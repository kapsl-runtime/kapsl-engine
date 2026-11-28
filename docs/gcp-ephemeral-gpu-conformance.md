# Ephemeral GCP GPU conformance runners

The managed-vLLM GPU conformance pipeline can create a fresh Google Compute
Engine VM for one GitHub Actions job and delete it when the job finishes. No
permanent self-hosted runner is required.

The implemented lifecycle is:

```text
official vMAJOR.MINOR.PATCH release tag
  -> Release Runtime Installers calls the reusable GPU workflow
  -> GitHub OIDC authenticates to a narrow GCP provisioner account
  -> a GitHub App creates a repository-scoped JIT runner registration
  -> one g2-standard-12 VM (one NVIDIA L4, 24 GiB VRAM) starts
  -> the VM validates driver 580+, Docker, CUDA 13, and the pinned runner
  -> the uniquely labelled runner accepts exactly one conformance job
  -> the JIT runner deregisters and the VM powers off
  -> an always-run hosted cleanup job deletes the VM and boot disk
```

Two independent fallbacks limit cost if GitHub is cancelled or unavailable:

- GCE deletes the VM after a hard five-hour maximum runtime.
- A scheduled workflow deletes any expired VM carrying the exact Kapsl
  ownership, purpose, repository, and expiry labels.

The VM has no Google service account, receives no GitHub App token, and adds no
inbound firewall rule. Only the single-use JIT configuration is copied into VM
metadata. Its unique label deliberately omits the generic `gpu` label, so an
older queued job cannot claim it. Each reusable workflow independently
revalidates the stable-release tag context before selecting a GPU runner.

## 1. Create the GCP project and quota

Select a project and a region with L4 availability. The examples use
`us-central1-a`; substitute a zone that has quota for:

- one NVIDIA L4 GPU;
- twelve G2 vCPUs;
- one 200 GiB balanced persistent disk.

Enable the required APIs:

```bash
export GCP_GPU_PROJECT_ID=your-gcp-project
export GCP_GPU_ZONE=us-central1-a

gcloud services enable \
  compute.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com \
  --project "$GCP_GPU_PROJECT_ID"
```

G2 availability varies by zone. Official release conformance always uses the
`STANDARD` provisioning model so a Spot preemption cannot invalidate release
evidence.

## 2. Pin an immutable NVIDIA 580 image

The startup contract expects Ubuntu 24.04 with a working NVIDIA 580-or-newer
driver. Resolve the current Google Deep Learning VM family once, review it,
then store the immutable image name rather than the mutable family:

```bash
image_name=$(gcloud compute images describe-from-family \
  common-cu129-ubuntu-2404-nvidia-580 \
  --project deeplearning-platform-release \
  --format='value(name)')
export GCP_GPU_RUNNER_IMAGE="projects/deeplearning-platform-release/global/images/$image_name"
printf '%s\n' "$GCP_GPU_RUNNER_IMAGE"
```

The VM bootstrap installs Docker and NVIDIA Container Toolkit, downloads
GitHub Actions Runner `2.337.0` with its pinned SHA-256 digest, pre-pulls the
same digest-pinned CUDA 13 image used by the conformance job, and refuses to
register if any GPU check fails.

## 3. Choose networking

The simplest setup uses the default VPC and an ephemeral external address. No
inbound rule is created by this workflow, and project SSH keys are blocked on
the VM. Configure:

```text
GCP_GPU_SUBNET=
GCP_GPU_EXTERNAL_IP=true
```

For a runner with no public address, create a dedicated subnet with Cloud NAT:

```bash
export GCP_GPU_REGION=${GCP_GPU_ZONE%-*}

gcloud compute networks create kapsl-ci \
  --project "$GCP_GPU_PROJECT_ID" \
  --subnet-mode custom
gcloud compute networks subnets create kapsl-gpu-runners \
  --project "$GCP_GPU_PROJECT_ID" \
  --network kapsl-ci \
  --region "$GCP_GPU_REGION" \
  --range 10.88.0.0/24
gcloud compute routers create kapsl-ci-nat-router \
  --project "$GCP_GPU_PROJECT_ID" \
  --network kapsl-ci \
  --region "$GCP_GPU_REGION"
gcloud compute routers nats create kapsl-ci-nat \
  --project "$GCP_GPU_PROJECT_ID" \
  --router kapsl-ci-nat-router \
  --router-region "$GCP_GPU_REGION" \
  --nat-all-subnet-ip-ranges \
  --auto-allocate-nat-external-ips
```

Then configure:

```text
GCP_GPU_SUBNET=projects/PROJECT_ID/regions/REGION/subnetworks/kapsl-gpu-runners
GCP_GPU_EXTERNAL_IP=false
```

No ingress firewall rule is needed. The VM only makes outbound HTTPS requests
to GitHub, package repositories, Hugging Face, and container registries.

## 4. Create the GCP provisioner identity

Create a keyless service account. It is used only by the hosted provisioning,
cleanup, and sweeper jobs; it is not attached to the GPU VM.

```bash
export GCP_GPU_PROVISIONER_NAME=kapsl-gpu-provisioner
export GCP_GPU_PROVISIONER_SERVICE_ACCOUNT="${GCP_GPU_PROVISIONER_NAME}@${GCP_GPU_PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create "$GCP_GPU_PROVISIONER_NAME" \
  --project "$GCP_GPU_PROJECT_ID" \
  --display-name 'Kapsl ephemeral GPU runner provisioner'
gcloud projects add-iam-policy-binding "$GCP_GPU_PROJECT_ID" \
  --member "serviceAccount:$GCP_GPU_PROVISIONER_SERVICE_ACCOUNT" \
  --role roles/compute.instanceAdmin.v1
```

For stricter production IAM, replace `roles/compute.instanceAdmin.v1` with a
custom role limited to instance create/get/list/delete, disk create/delete,
subnetwork use, and image read. The workflow never attaches a service account
to the VM.

## 5. Configure GitHub OIDC to GCP

Create one Workload Identity Pool/provider restricted to this repository's
`gcp-gpu-conformance` environment:

```bash
export GCP_PROJECT_NUMBER=$(gcloud projects describe "$GCP_GPU_PROJECT_ID" --format='value(projectNumber)')
export GCP_WIF_POOL=github-actions
export GCP_WIF_PROVIDER=kapsl-engine

gcloud iam workload-identity-pools create "$GCP_WIF_POOL" \
  --project "$GCP_GPU_PROJECT_ID" \
  --location global \
  --display-name 'GitHub Actions'
gcloud iam workload-identity-pools providers create-oidc "$GCP_WIF_PROVIDER" \
  --project "$GCP_GPU_PROJECT_ID" \
  --location global \
  --workload-identity-pool "$GCP_WIF_POOL" \
  --issuer-uri https://token.actions.githubusercontent.com \
  --attribute-mapping 'google.subject=assertion.sub,attribute.repository=assertion.repository' \
  --attribute-condition "assertion.repository == 'kapsl-runtime/kapsl-engine' && assertion.sub == 'repo:kapsl-runtime/kapsl-engine:environment:gcp-gpu-conformance'"

export GCP_WORKLOAD_IDENTITY_PROVIDER=$(gcloud iam workload-identity-pools providers describe \
  "$GCP_WIF_PROVIDER" \
  --project "$GCP_GPU_PROJECT_ID" \
  --location global \
  --workload-identity-pool "$GCP_WIF_POOL" \
  --format='value(name)')

gcloud iam service-accounts add-iam-policy-binding \
  "$GCP_GPU_PROVISIONER_SERVICE_ACCOUNT" \
  --project "$GCP_GPU_PROJECT_ID" \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/$GCP_PROJECT_NUMBER/locations/global/workloadIdentityPools/$GCP_WIF_POOL/attribute.repository/kapsl-runtime/kapsl-engine"
```

This uses short-lived GitHub OIDC credentials; do not create or upload a GCP
service-account key.

## 6. Create the GitHub runner App

Create a private GitHub App owned by `kapsl-runtime` with:

- repository permission **Administration: Read and write**;
- no organization permissions;
- webhooks disabled;
- installation limited to `kapsl-engine`.

Generate one private key. The App token exists only in the hosted provisioner
and cleanup jobs. The GPU VM receives the single-use JIT runner configuration,
not the App key or token.

## 7. Configure the protected GitHub environment

Create the environment `gcp-gpu-conformance` with custom deployment policies
that allow the `main` branch and `v*` tags. `main` is required for the scheduled
expired-runner sweeper; release tags are required for conformance. Do not add a
required-reviewer rule if cleanup must be completely unattended: GitHub
evaluates environment protection for each job, including the always-run
deletion job and scheduled sweeper. The workflow itself accepts only an exact
stable `vMAJOR.MINOR.PATCH` tag and rejects branches, manual dispatches, beta
tags, and release candidates before provisioning.

Set these environment variables:

| Variable | Value |
| --- | --- |
| `GCP_GPU_PROJECT_ID` | GCP project ID |
| `GCP_GPU_ZONE` | L4-capable zone, such as `us-central1-a` |
| `GCP_GPU_RUNNER_IMAGE` | Exact `projects/.../global/images/...` value from step 2 |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | Full provider name from step 5 |
| `GCP_GPU_PROVISIONER_SERVICE_ACCOUNT` | Provisioner service-account email |
| `GPU_RUNNER_GITHUB_APP_ID` | Numeric GitHub App ID |
| `GCP_GPU_RUNNER_GROUP_ID` | Repository runner group ID; normally `1` |
| `GCP_GPU_SUBNET` | Empty for default VPC, or the full subnet resource path |
| `GCP_GPU_EXTERNAL_IP` | `true` for an ephemeral address, or `false` with Cloud NAT |

Set this environment secret:

| Secret | Value |
| --- | --- |
| `GPU_RUNNER_GITHUB_APP_PRIVATE_KEY` | Entire PEM private key downloaded from the App |

The vLLM connector that a release packages and this job certifies is not a
repository setting. `.github/vllm-integration.lock` holds the exact lowercase
40-character `kapsl-integrations` commit the connector is built from, so the
engine commit being released selects it. Promote a new connector with a pull
request that changes that file together with the connector version the engine
pins; branch names and floating tags are rejected. The repository variable
`KAPSL_VLLM_SDK_REF`, which used to name a `kapsl-sdk` commit, is no longer
read.

After creating the environment in the repository settings and saving the App
private key as `github-app-private-key.pem`, the non-interactive configuration
is:

```bash
export GPU_RUNNER_GITHUB_APP_ID=123456

gh api --method PUT \
  repos/kapsl-runtime/kapsl-engine/environments/gcp-gpu-conformance >/dev/null
gh variable set GCP_GPU_PROJECT_ID --repo kapsl-runtime/kapsl-engine \
  --env gcp-gpu-conformance --body "$GCP_GPU_PROJECT_ID"
gh variable set GCP_GPU_ZONE --repo kapsl-runtime/kapsl-engine \
  --env gcp-gpu-conformance --body "$GCP_GPU_ZONE"
gh variable set GCP_GPU_RUNNER_IMAGE --repo kapsl-runtime/kapsl-engine \
  --env gcp-gpu-conformance --body "$GCP_GPU_RUNNER_IMAGE"
gh variable set GCP_WORKLOAD_IDENTITY_PROVIDER --repo kapsl-runtime/kapsl-engine \
  --env gcp-gpu-conformance --body "$GCP_WORKLOAD_IDENTITY_PROVIDER"
gh variable set GCP_GPU_PROVISIONER_SERVICE_ACCOUNT --repo kapsl-runtime/kapsl-engine \
  --env gcp-gpu-conformance --body "$GCP_GPU_PROVISIONER_SERVICE_ACCOUNT"
gh variable set GPU_RUNNER_GITHUB_APP_ID --repo kapsl-runtime/kapsl-engine \
  --env gcp-gpu-conformance --body "$GPU_RUNNER_GITHUB_APP_ID"
gh variable set GCP_GPU_RUNNER_GROUP_ID --repo kapsl-runtime/kapsl-engine \
  --env gcp-gpu-conformance --body 1
gh variable set GCP_GPU_EXTERNAL_IP --repo kapsl-runtime/kapsl-engine \
  --env gcp-gpu-conformance --body true
gh secret set GPU_RUNNER_GITHUB_APP_PRIVATE_KEY \
  --repo kapsl-runtime/kapsl-engine \
  --env gcp-gpu-conformance \
  < github-app-private-key.pem
```

For the private-subnet option, also set `GCP_GPU_SUBNET` to the full resource
path and change `GCP_GPU_EXTERNAL_IP` to `false`.

The repository's built-in `GITHUB_TOKEN` cannot create JIT runner
registrations because that endpoint requires repository Administration write;
that is why the narrowly installed App is required.

## 8. Run conformance

There is no manual GPU trigger. After the separate `develop` to `main` release
PR is merged, pushing the official stable `vMAJOR.MINOR.PATCH` tag starts
`release-runtime-installers.yml`, which calls the GPU workflow exactly once.
The beta installer workflow does not call it. The GPU entry-point workflows
expose only `workflow_call`, and each repeats the stable-tag authorization
before a real GPU runner can be selected.

The hosted preparation job waits until the exact unique runner is online before
it releases the conformance job, preventing a silent 24-hour self-hosted-runner
queue. If you deliberately configure environment reviewers, approve every
pending lifecycle job promptly, including cleanup; otherwise rely on branch
restrictions for an unattended run.

The official run always uses `STANDARD` provisioning.

## Diagnostics and cleanup

The workflow uploads the VM serial log before deletion. Bootstrap failures are
published as `gcp-gpu-runner-bootstrap-*`; completed runs publish
`gcp-gpu-runner-*`. Conformance evidence remains in the artifacts emitted by
the reusable vLLM workflow.

List managed instances without deleting anything:

```bash
gcloud compute instances list \
  --project "$GCP_GPU_PROJECT_ID" \
  --filter='labels.managed-by=kapsl-gha AND labels.purpose=vllm-conformance'
```

The checked-in sweeper can be run manually after authenticating with the
provisioner identity:

```bash
python3 .github/scripts/gcp_ephemeral_gpu_runner.py sweep \
  --project "$GCP_GPU_PROJECT_ID" \
  --repository kapsl-runtime/kapsl-engine
```

It only deletes an instance when every ownership label matches, the expiry is
in the past, and both the instance name and zone pass strict validation. The
normal cleanup path deletes only the deterministic instance for the current
GitHub run.

## When GCP has no L4 capacity

Provisioning fails with `ZONE_RESOURCE_POOL_EXHAUSTED` (`STOCKOUT`) when the
zone has no L4 machine. Nothing was built wrong, and the run keeps its signed
artifacts. In order of preference:

1. Point `GCP_GPU_ZONE` at another L4 zone of the subnet's region, then re-run
   only the failed jobs. The run reuses its artifacts.
2. Wait and re-run the failed jobs later.
3. Qualify that run's CUDA artifact by hand and waive the automated job for
   that run only.

### Waiving the GCP job for one run

1. Download the run's `runtime-cuda-linux-x86_64` artifact on a GPU host and
   cover what the automated job covers: the signed ORT CUDA 12 and TensorRT 10
   packs, the managed vLLM shared pool, and llama.cpp next to vLLM. Record the
   artifact digest that the artifacts API reports
   (`gh api repos/kapsl-runtime/kapsl-engine/actions/runs/<run>/artifacts --jq '.artifacts[] | {name, digest}'`)
   and check that the downloaded zip has the same SHA-256.
2. Post the results in this repository (an issue or PR comment).
3. Merge `.github/stable-gpu-gate-waiver.json` into `main`:

   ```json
   {
     "schema_version": 1,
     "release_tag": "v1.2.3",
     "run_id": 46080324165,
     "candidate": {
       "artifact": "runtime-cuda-linux-x86_64",
       "digest": "sha256:<64 hex digits>"
     },
     "reason": "Why the automated job could not run.",
     "manual_qualification": {
       "performed_on": "2026-11-28",
       "hardware": "1x RTX 3090 24 GB, driver 595",
       "evidence": "https://github.com/kapsl-runtime/kapsl-engine/..."
     }
   }
   ```

4. Re-run the failed jobs of that release run. The GCP job is tried again. If
   it still fails, `release-conformance-gate` reads the waiver from `main` and
   `stable_gpu_gate_waiver.py` accepts it only when the tag, the run ID and the
   digest of that run's CUDA artifact all match. Then the run publishes the
   artifacts that were qualified.

- The waiver is read when the gate runs, not from the tag, so no rebuild is
  needed, and it can only publish the exact artifact that was qualified.
- Only the GPU job can be waived. Infrastructure authentication, signing and
  CPU parity must still pass.
- Anything malformed or mismatched keeps publication blocked.
- The run still ends as failed, because the GCP job failed. The gate's warning
  names the waiver, so the record stays in the run.

Official references:

- [GitHub JIT runner REST API](https://docs.github.com/en/rest/actions/self-hosted-runners)
- [GitHub ephemeral runner guidance](https://docs.github.com/en/actions/reference/runners/self-hosted-runners)
- [Google GitHub OIDC authentication](https://github.com/google-github-actions/auth)
- [GCE maximum VM runtime](https://cloud.google.com/compute/docs/instances/limit-vm-runtime)
- [Google Deep Learning VM images](https://cloud.google.com/deep-learning-vm/docs/images)
