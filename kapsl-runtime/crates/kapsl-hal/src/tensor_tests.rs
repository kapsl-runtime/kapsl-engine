use crate::tensor::{TensorView, TensorViewMut};

#[test]
fn tensor_view_reports_len_and_shape() {
    let data = vec![1u8, 2, 3, 4];
    let shape = [2usize, 2];
    let view = TensorView::new(&data, &shape);

    assert_eq!(view.len(), 4);
    assert!(!view.is_empty());
    assert_eq!(view.shape, &shape);
}

#[test]
fn tensor_view_mut_reports_len_and_allows_mutation() {
    let mut data = vec![0i32, 1, 2, 3];
    let shape = [4usize];
    {
        let view = TensorViewMut::new(&mut data, &shape);
        assert_eq!(view.len(), 4);
        assert!(!view.is_empty());
        assert_eq!(view.shape, &shape);
    }
    data[0] = 42;
    assert_eq!(data[0], 42);
}

#[test]
fn tensor_view_empty_is_empty() {
    let data: Vec<f32> = Vec::new();
    let shape = [0usize];
    let view = TensorView::new(&data, &shape);

    assert_eq!(view.len(), 0);
    assert!(view.is_empty());
}
