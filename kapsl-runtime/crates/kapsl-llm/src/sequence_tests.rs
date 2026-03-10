#[cfg(test)]
mod tests {
    use super::super::{FinishReason, SamplingParams, Sequence, SequenceGroup, SequenceStatus};
    use tokio::sync::mpsc;

    #[test]
    fn sequence_appends_tokens_and_tracks_length() {
        let mut seq = Sequence::new(42, "hello".to_string(), vec![1, 2]);
        assert_eq!(seq.sequence_id, 42);
        assert_eq!(seq.get_len(), 2);
        assert!(!seq.is_finished());
        assert_eq!(seq.cumulative_logprob, 0.0);

        seq.append_token_id(3, -0.5);
        seq.append_token_id(4, -0.25);
        assert_eq!(seq.output_token_ids, vec![3, 4]);
        assert_eq!(seq.get_len(), 4);
        assert!((seq.cumulative_logprob + 0.75).abs() < f32::EPSILON);

        seq.status = SequenceStatus::Finished(FinishReason::Stop);
        assert!(seq.is_finished());
    }

    #[test]
    fn sequence_group_finishes_when_all_sequences_finish() {
        let (tx, _rx) = mpsc::channel(1);
        let sampling = SamplingParams {
            max_tokens: 4,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            stop_token_ids: vec![2],
            repetition_penalty: 1.1,
            seed: None,
        };
        let mut group = SequenceGroup::new(
            "req".to_string(),
            None,
            "prompt".to_string(),
            vec![10, 11],
            sampling,
            None,
            tx,
        );

        assert_eq!(group.sequences.len(), 1);
        assert!(!group.is_finished());

        let seq_arc = group.get_seqs().pop().expect("sequence");
        let (old_status, new_status) = {
            let mut seq = seq_arc.lock().unwrap();
            let old_status = seq.status;
            seq.status = SequenceStatus::Finished(FinishReason::Stop);
            (old_status, seq.status)
        };
        group.update_seq_status(old_status, new_status);

        assert!(group.is_finished());
    }
}
