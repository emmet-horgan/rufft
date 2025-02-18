searchState.loadedDescShard("rufft", 0, "Rufft is a pure rust signal processing library which …\nA complex number in Cartesian form.\nReturn Euler’s number.\nReturn <code>1.0 / π</code>.\nReturn <code>1.0 / sqrt(2.0)</code>.\nReturn <code>2.0 / π</code>.\nReturn <code>2.0 / sqrt(π)</code>.\nReturn <code>π / 2.0</code>.\nReturn <code>π / 3.0</code>.\nReturn <code>π / 4.0</code>.\nReturn <code>π / 6.0</code>.\nReturn <code>π / 8.0</code>.\nGeneric trait for floating point numbers\nA constant <code>Complex</code> <em>i</em>, the imaginary unit.\nReturn <code>ln(10.0)</code>.\nReturn <code>ln(2.0)</code>.\nReturn <code>log10(2.0)</code>.\nReturn <code>log10(e)</code>.\nReturn <code>log2(10.0)</code>.\nReturn <code>log2(e)</code>.\nA constant <code>Complex</code> 1.\nReturn Archimedes’ constant <code>π</code>.\nReturn <code>sqrt(2.0)</code>.\nReturn the full circle constant <code>τ</code>.\nA constant <code>Complex</code> 0.\nComputes the absolute value of <code>self</code>. Returns <code>Float::nan()</code> …\nThe positive difference of two numbers.\nComputes the arccosine of a number. Return value is in …\nComputes the principal value of the inverse cosine of <code>self</code>.\nInverse hyperbolic cosine function.\nComputes the principal value of inverse hyperbolic cosine …\nCalculate the principal Arg of self.\nComputes the arcsine of a number. Return value is in …\nComputes the principal value of the inverse sine of <code>self</code>.\nInverse hyperbolic sine function.\nComputes the principal value of inverse hyperbolic sine of …\nComputes the arctangent of a number. Return value is in …\nComputes the principal value of the inverse tangent of <code>self</code>…\nComputes the four quadrant arctangent of <code>self</code> (<code>y</code>) and <code>other</code>…\nInverse hyperbolic tangent function.\nComputes the principal value of inverse hyperbolic tangent …\nTake the cubic root of a number.\nComputes the principal value of the cube root of <code>self</code>.\nReturns the smallest integer greater than or equal to a …\nCreate a new Complex with a given phase: <code>exp(i * phase)</code>. …\nClamps a value between a min and max.\nReturns the floating point category of the number. If only …\nReturns the complex conjugate. i.e. <code>re - i im</code>\nReturns a number composed of the magnitude of <code>self</code> and the …\nComputes the cosine of a number (in radians).\nComputes the cosine of <code>self</code>.\nHyperbolic cosine function.\nComputes the hyperbolic cosine of <code>self</code>.\nReturns epsilon, a small positive value.\nReturns <code>e^(self)</code>, (the exponential function).\nComputes <code>e^(self)</code>, where <code>e</code> is the base of the natural …\nReturns <code>2^(self)</code>.\nComputes <code>2^(self)</code>.\nReturns <code>e^(self) - 1</code> in a way that is accurate even if the …\nRaises a floating point number to the complex power <code>self</code>.\nReturns <code>self/other</code> using floating-point operations.\nThe <code>fft</code> module itself contains some basic functions ;ole …\nReturns <code>1/self</code> using floating-point operations.\nReturns the largest integer less than or equal to a number.\nReturns the fractional part of a number.\nReturns the argument unchanged.\nConvert a polar representation into a complex number.\nParses <code>a +/- bi</code>; <code>ai +/- b</code>; <code>a</code>; or <code>bi</code> where <code>a</code> and <code>b</code> are of …\nParses <code>a +/- bi</code>; <code>ai +/- b</code>; <code>a</code>; or <code>bi</code> where <code>a</code> and <code>b</code> are of …\nCalculate the length of the hypotenuse of a right-angle …\nReturns the imaginary unit.\nImaginary portion of the complex number\nReturns the infinite value.\nReturns the mantissa, base 2 exponent, and sign as …\nCalls <code>U::from(self)</code>.\nReturns <code>1/self</code>\nReturns <code>true</code> if this number is neither infinite nor <code>NaN</code>.\nChecks if the given complex number is finite\nReturns <code>true</code> if this value is positive infinity or …\nChecks if the given complex number is infinite\nReturns <code>true</code> if this value is <code>NaN</code> and false otherwise.\nChecks if the given complex number is NaN\nReturns <code>true</code> if the number is neither zero, infinite, …\nChecks if the given complex number is normal\nReturns <code>true</code> if <code>self</code> is negative, including <code>-0.0</code>, …\nReturns <code>true</code> if <code>self</code> is positive, including <code>+0.0</code>, …\nReturns <code>true</code> if the number is subnormal.\nSignal processing functions which operate on <code>Iterable</code> …\nReturns the L1 norm <code>|re| + |im|</code> – the Manhattan distance …\nReturns the natural logarithm of the number.\nComputes the principal value of natural logarithm of <code>self</code>.\nReturns <code>ln(1+n)</code> (natural logarithm) more accurately than if\nReturns the logarithm of the number with respect to an …\nReturns the logarithm of <code>self</code> with respect to an arbitrary …\nReturns the base 10 logarithm of the number.\nComputes the principal value of log base 10 of <code>self</code>.\nReturns the base 2 logarithm of the number.\nComputes the principal value of log base 2 of <code>self</code>.\nReturns the maximum of the two numbers.\nReturns the largest finite value that this type can …\nReturns the minimum of the two numbers.\nReturns the smallest positive, normalized value that this …\nReturns the smallest finite value that this type can …\nFused multiply-add. Computes <code>(self * a) + b</code> with only one …\nReturns the <code>NaN</code> value.\nReturns the negative infinite value.\nReturns <code>-0.0</code>.\nCreate a new <code>Complex</code>\nCalculate |self|\nReturns the square of the norm (since <code>T</code> doesn’t …\nRaises <code>self</code> to a complex power.\nRaise a number to a floating point power.\nRaises <code>self</code> to a floating point power.\nRaise a number to an integer power.\nRaises <code>self</code> to a signed integer power.\nRaises <code>self</code> to an unsigned integer power.\nReal portion of the complex number\nTake the reciprocal (inverse) of a number, <code>1/x</code>.\nReturns the nearest integer to a number. Round half-way …\nMultiplies <code>self</code> by the scalar <code>t</code>.\nReturns a number that represents the sign of <code>self</code>.\nComputes the sine of a number (in radians).\nComputes the sine of <code>self</code>.\nSimultaneously computes the sine and cosine of the number, …\nHyperbolic sine function.\nComputes the hyperbolic sine of <code>self</code>.\nTake the square root of a number.\nComputes the principal value of the square root of <code>self</code>.\nComputes the tangent of a number (in radians).\nComputes the tangent of <code>self</code>.\nHyperbolic tangent function.\nComputes the hyperbolic tangent of <code>self</code>.\nConverts radians to degrees.\nConvert to polar form (r, theta), such that …\nConverts degrees to radians.\nReturn the integer part of a number.\nDivides <code>self</code> by the scalar <code>t</code>.\nComputes the discrete fourier tranform on the real valued …\nCompute the discrete time fourier transform of the real …\nComputes the frequency values associated fft based on <code>n</code> …\nComputes the frequency values associated fft based on <code>n</code> …\nComputes the inverse discrete fourier transform of the …\nWraps an angle in radians to the range (-π, π].\nCompute the discrete fourier transform on the complex …\nComputes the inverse discrete fourier transform on the …\nComputes the cooley-tukey fast fourier transform of the …\nCompute the inverse cooley-tukey fast fourier transform of …\nComputes the cooley-tukey fast fourier transform on the …\nComputes the chirp-z fast fourier transform of the real …\nClone and pad the real valued input collection with the …\nPad the real valued input collection inplace with the …\nClone and pad the real valued input collection with …\nPad the real valued input collection in place with …\nClone and zero pad the real valued input collection by the …\nZero pad the real valued input collection by the length <code>n</code>. …\nClone and zero pad the real valued input collection to the …\nZero pad the real valued input collection inplace to the …\nClone and pad the complex valued input collection with the …\nPad the complex valued input collection inplace with the …\nClone and pad the complex valued input collection with …\nPad the complex valued input collection in place with the …\nClone and zero pad the complex valued input collection by …\nZero pad the complex valued input collection by the length …\nClone and zero pad the complex valued input collection to …\nZero pad the real valued input collection inplace to the …\nExtendable iterable trait that can be implemented on …\nTrait containing <code>fft</code> method which computes the <code>fft</code> of the …\nThe item produced by the collection’s iterator. For …\nIterable trait to encapsulate collection types which have …\nThe iterator produced by the collection\nThe owned collection item. For example <code>f64</code> for <code>Vec&lt;f64&gt;</code>\nExtend the collection from a slice of …\nCreate an iterator from the collection over <code>Self::Item</code> …\nReturn the length of the collection. Default …\nPush a <code>&lt;Self as Iterable&gt;::OwnedItem</code> type to the collection")