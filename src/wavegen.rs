pub mod sine;
pub mod sinc;

macro_rules! impl_traits {
    ($($name:ident),*) => {
        $(
            impl <F, C> FromIterator<F> for $name<F, C>
            where 
                for<'c> C: Iterable<OwnedItem = F, Item<'c> = &'c F>,
                F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>,
                usize: AsPrimitive<F>
            {
                fn from_iter<I: IntoIterator<Item = F>>(iter: I) -> Self {
                    Self (
                        iter.into_iter().collect()
                    )
                }

            }

            impl<F, C> Iterable for $name<F, C>
            where 
                for<'c> C: Iterable<OwnedItem = F, Item<'c> = &'c F>,
                F: Float + FloatConst + NumAssign + 'static + AsPrimitive<usize>,
                usize: AsPrimitive<F>
            {
                type OwnedItem = F;
                type Item<'c> = &'c F;
                type Iterator<'c> = <C as Iterable>::Iterator<'c>;
            
                fn iter<'c>(&'c self) -> Self::Iterator<'c> {
                    self.0.iter()
                }
            }
        )*
    };
}

pub(crate) use impl_traits;
