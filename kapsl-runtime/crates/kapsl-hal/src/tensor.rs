use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct TensorView<'a, T> {
    pub data: &'a [T],
    pub shape: &'a [usize],
}

impl<'a, T> TensorView<'a, T> {
    pub fn new(data: &'a [T], shape: &'a [usize]) -> Self {
        Self { data, shape }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[derive(Debug)]
pub struct TensorViewMut<'a, T> {
    pub data: &'a mut [T],
    pub shape: &'a [usize],
}

impl<'a, T> TensorViewMut<'a, T> {
    pub fn new(data: &'a mut [T], shape: &'a [usize]) -> Self {
        Self { data, shape }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
