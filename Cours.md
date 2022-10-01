#	Functional Programming 

{:toc}



## Week1  : 

### call by value vs call by name : 

**Call-by-value :** evaluate the parameters *then* execute the function on these values. 

e.g : 

````
sumOfSquares(3, 2+2) 
sumOfSquares(3, 4) // evaluated 2+2 BEFORE executing sumOfSquares
square(3) + square(4)
3 * 3 + square(4)
9 + square(4)
9 + 4 * 4
9 + 16
25
````

Call-by-value has the advantage that it evaluates every function argument
only once.

**Call-by-name :** Apply the function to unreduced arguments. 

e.g : 

```
sumOfSquares(3, 2+2)
square(3) + square(2+2) // executing sumeOfSquare before evaluating 2+2 
3 * 3 + square(2+2) 
9 + square(2+2)
9 + (2+2) * (2+2)
9 + 4 * (2+2)
9 + 4 * 4
```

Call-by-name has the advantage that a function argument is not evaluated
if the corresponding parameter is unused in the evaluation of the function
body.

e.g :`` ``

```scala
def loop : Int = loop 
def f(a:Int , b : Int ) : Int = a // Call-by-value 
f(1,loop) // does not terminate 
def f(a:Int, b : => Int) : Int = a // Call-by-Name ( => )
f(1,loop) // terminates 
```



Both strategies reduce to the same final values as long as
▶ the reduced expression consists of pure functions, and
▶ both evaluations terminate.

### Conditionals : 

```scala
def abs(x: Int) = if x >= 0 then x else -x
```

``if-then-else``  in  Scala is an expression (has a value) not a statement .

### Blocks , Scopes and first functional program : 

Sqrt with Newton's method : 

```scala
def sqrt(x: Double) = {
    def sqrtIter(guess: Double, x: Double): Double =
        if isGoodEnough(guess, x) then guess
        else sqrtIter(improve(guess, x), x)
    def improve(guess: Double, x: Double) =(guess + x / guess) /     	
    def isGoodEnough(guess: Double, x: Double) =abs(square(guess) - x) < 0.001
    	
    sqrtIter(1.0, x)
}
```

``sqrtIter`` , ``improve`` and ``isGoodEnough`` should not be outside sqrt because : 

* they pollute the global scope. 
* they should not be accessible to the user of the sqrt code given they are specific to sqrt . 

the {} could be removed for more clarity. 

### Tail recursion : 

```scala
def gcd(a: Int, b: Int): Int =if b == 0 then a else gcd(b, a % b)
```

<img src="C:\Users\XPS13\AppData\Roaming\Typora\typora-user-images\image-20220922085907893.png" alt="image-20220922085907893" style="zoom: 67%;" />

`gcd(14,21) = gcd(21,14) = gcd(14,7) = gcd(..,...) = ... = 7 ` : flat structure, `gcd` only calls `gcd`=>  **tail recursion**. 

```scala
def factorial(n: Int): Int = if n == 0 then 1 else n * factorial(n - 1)
```

![image-20220922090107614](C:\Users\XPS13\AppData\Roaming\Typora\typora-user-images\image-20220922090107614.png) 

`factorial` calls factorial *and multiplies* , nested structure `=>` **not tail recursion**

**To tell the compiler than a function is tail recursive: **

```scala
import scala.annotation.tailrec
@tailrec
def gcd(a: Int, b: Int): Int = ...
```





## Week 2 : 

**Higher order functions :** 

*Def :* Functions that take other functions as parameters and return functions. 

example : 

```scala
def sum( f : Int => Int )( a : Int , b : Int ) : Int 
```

* takes as argument : the function `f : Int => Int`.
* returns : a function that has 2 arguments `a:Int,b:Int` and returns `Int`  

**Function types : ** `A=>B` is the type of a function that takes an argument of type A and returns a result of type B.  e.g : `Int=>Int` maps integers to integers. 

**Anonymous Functions :** 

```scala
(x: Int, y: Int) => x + y
```

Use case : 

```scala
def sumCubes(a: Int, b: Int) = sum(x => x * x * x, a, b) // type of function is inferred by the compiler 
```

### Currying : 

Suppose we have this header for `sum` : `def sum(f : Int=>Int , a : Int , b : Int)` 

```scala
def sumInts(a: Int, b: Int) = sum(x => x, a, b)
def sumCubes(a: Int, b: Int) = sum(x => x * x * x, a, b)
def sumFactorials(a: Int, b: Int) = sum(fact, a, b)
```

Notice that a and b get passed unchanged from `sumInts` to `sum` . 

*Q :* Can we do it shorter ?

```scala
def sum(f: Int => Int): (Int, Int) => Int =
    def sumF(a: Int, b: Int): Int =
        if a > b then 0
        else f(a) + sumF(a + 1, b)
sumF

def sumInts = sum(x => x)
def sumCubes = sum(x => x * x * x)
def sumFactorials = sum(fact)
```

  `sum` is now a function that takes 1 argument (`f`) and returns a function that takes 2 arguments `a` and `b`.

We can still do shorter : 

```scala
def sum(f: Int => Int)(a: Int, b: Int): Int =
	if a > b then 0 else f(a) + sum(f)(a + 1, b)
```

*Q :* What is the type of `sum` ? 

*A :* ` (Int=>Int) => ((Int,Int)=>Int) `  equivalent to 

`(Int=>Int) => (Int,Int)=>Int`

Note that function types **associate** **to the right** so it is equivalent to : 

`Int => Int => Int` is equivalent to`Int => (Int => Int)`



### Classes : 

we can define classes in Scala : 

```scala
class Rational(x: Int, y: Int):
    def numer = x // numer and denom here are functions (recalculated each call)
    def denom = y

	// adding operations 
	def addRational(r: Rational, s: Rational): 
		Rational = Rational(r.numer * s.denom + s.numer * r.denom,r.denom * 			s.denom)
    // overriding toString 
	override def toString = s”$numer/$denom” // s means formatted string 
	// what is after a $ is evaluated  

val x = Rational(1, 2) // x: Rational = Rational@2abe0e27
x.numer // 1
x.denom // 2
val y = Rational(5, 7)
val z = Rational(3, 2)
x.add(y).add(z) 

```

Classes are very similar to Java. 

Suppose that `numer` and `denom` need more calculations , like the following : 

```scala
class Rational(x: Int, y: Int):
	private def gcd(a: Int, b: Int): Int = 
		if b == 0 then a else gcd(b, a % b)
    def numer = x / gcd(x, y) // these are recalcaled at each call 
    def denom = y / gcd(x, y) // BAD IDEA
	
	// BETTER IDEA : VARIABLES : 
	val numer = x / gcd(x, y)
	val denom = y / gcd(x, y)

```

It is possible to use `this.(..)` like is Java. 

### Substitution and Extensions : 

#### 1. Substitution : 

Suppose that we have a class definition :

`class C(x1, ..., xm){ ... def f(y1, ..., yn) = b ... }` 

 *Q :* How is the following expression evaluated ? `C(v1, ..., vm).f(w1, ..., wn)`

*A :*  Three substitutions happen written : 

`[w1/y1, ..., wn/yn][v1/x1, ..., vm/xm][C(v1, ..., vm)/this] b`

* Substitution of  `y1, ..., yn` of  f by the arguments `w1, ..., wn` . 

* Substitution of the   `x1, ..., xm` of the class C by the class arguments `v1, ..., vm`. 

* Substitution of the self reference `this` by the value of the object `C(v1, ..., vn)`.

#### 2. Extension methods (1) : 

Having to define all methods that belong to a class inside the class itself can lead to very large classes, and is not very modular.

Methods that do not need to access the internals of a class can alternatively be defined as extension methods. 

For instance, we can add min and abs methods to class Rational like this: 

```scala
extension (r: Rational): 
	def min(s: Rational): Boolean = if s.less(r) then s else r 
	def abs: Rational = Rational(r.numer.abs, r.denom)
```

* Extension methods **CANNOT** access internals (`private`) or `this.(..)`. 

#### 3. Operators : 

▶ We write `x + y`, if x and y are integers, but 

▶ We write `r.add(s)` if r and s are rational numbers. 

In Scala, we can eliminate this difference with **operators**. 

```scala
extension (x: Rational):
	def + (y: Rational): Rational = x.add(y)
```

**Precedence rules : **

![image-20220925091957252](C:\Users\XPS13\AppData\Roaming\Typora\typora-user-images\image-20220925091957252.png)

*(all other special characters)* having the **highest priority**. 

## Week3 : 

### Abstract classes : 

```scala
abstract class IntSet:
    def incl(x: Int): IntSet  // ABSTRACT MEMBERS : no implementation provided
    def contains(x: Int): Boolean 
// incl returns the union of {x} and this set 
```

Like in Java, abstract classes cannot be instantiated. 

We will try to implement a set as a binary tree. 

A set can either be : 

1. A tree for the **empty set**. 
2. A tree consisting of **one** integer with two subtrees. <img src="C:\Users\XPS13\AppData\Roaming\Typora\typora-user-images\image-20221001174610442.png" alt="image-20221001174610442" style="zoom:33%;" /> 

```scala
class Empty() extends IntSet:
	def contains(x: Int): Boolean = false
	def incl(x: Int): IntSet = NonEmpty(x, Empty(), Empty()) 

class NonEmpty(elem: Int, left: IntSet, right: IntSet) extends IntSet:
        def contains(x: Int): Boolean =
            if x < elem then left.contains(x)
            else if x > elem then right.contains(x) else true
		
        def incl(x: Int): IntSet =
        if x < elem then NonEmpty(elem, left.incl(x), right)
        else if x > elem then NonEmpty(elem, left, right.incl(x)) 
	else this
end NonEmpty
```

1. **Persistence :** 

   Note that we the sets are immutable, we always return a new tree while **reusing some subtrees**.(blue one in the following e.g)<img src="C:\Users\XPS13\AppData\Roaming\Typora\typora-user-images\image-20221001175338079.png" alt="image-20221001175338079" style="zoom:30%;" />

    A data structure that is creating by maintaining the old one is called **persistent**.

2. **Dynamic binding :**

   ```scala
   class Empty() extends IntSet:
   	... 
   	def union(other : Inset): Intset = other 
   class NonEmpty((elem: Int, left: IntSet, right: IntSet)) extends Inset : 
   	... 
   	def union(other : Inset ): Intset = l.union(r).union(that).incl(elem)
   
   // Why does this terminate ? Union is called with strictly smaller sets each time so union( "an empty set " ) will be called at some point.   
   ```

    Note that a call to `union` doesn't execute the same function if s is `Empty` or `NonEmpty`. That is called Dynamic **binding**. ( Polymorphism )

### Object definition : 

In the `IntSet` example, one could argue that there is really **only a single** empty `IntSet`. 

This defines a **singleton** object named Empty. No other Empty instance can be (or needs to be) created.

```scala
 object Empty extends IntSet: 
	def contains(x: Int): Boolean = false
	def incl(x: Int): IntSet = NonEmpty(x, Empty, Empty) end Empty
```

**Companion object :** 

If a class and object with the same name are given in the same sourcefile, we call them companions. Example:

```scala

class IntSet ...
object IntSet:
	def singleton(x: Int) = NonEmpty(x, Empty, Empty)
```

Similar to Java, static nested class. 

How is naming a class and an object the same name possible  ? Scala has **two** namespaces , one for objects and the other for classes. 

### Programs :

Like Java , scale source files can have a main methods  

```scala
object Hello:
	def main(args: Array[String]): Unit = println(”hello world!”)
// written Shorter as the following : 
@main def birthday(name: String, age: Int) =
println(s”Happy birthday, $name! $age years old already!”)
```

to run : `scala Hello`

### Traits : 

Java `Interface` but stronger :they can have parameters and can contain fields and concrete methods.

````scala
trait Planar:
    def height: Int
    def width: Int
    def surface = height * width
class Square extends Shape, Planar, Movable ...
````

![image-20221001203334184](C:\Users\XPS13\AppData\Roaming\Typora\typora-user-images\image-20221001203334184.png)

`- - -> `: can be converted (viewed) as 

### Cons List :

A fundamental data structure in many functional languages is the **immutable linked list.** 

It is constructed from two building blocks: 

1. `Nil` :the empty list 
2. `Cons`:a cell containing an element and the remainder of the list

```scala
trait IntList ...
class Cons(val head: Int, val tail: IntList) extends IntList ...
// NOTE THE USE OF val 
// this defines at the same time a parameter and a field of a class 
class Nil() extends IntList ...
```

A `IntList` is either a `Nil()` or a `Cons(x,xs)`. 

#### A complete definition with **Generics** : 

```scala
trait List[T]:
    def isEmpty: Boolean
    def head: T
    def tail: List[T]
class Cons[T](val head: T, val tail: List[T]) extends List[T]:
	def isEmpty = false
class Nil[T] extends List[T]:
    def isEmpty = true
    def head = throw new NoSuchElementException(”Nil.head”)
    def tail = throw new NoSuchElementException(”Nil.tail”)
```

#### Generic functions :

```scala
def singleton[T](elem: T) = Cons[T](elem, Nil[T])
// We can then write:
singleton[Int](1)
singleton[Boolean](true)
```

### Pure Object Orientation : 

*Def :* A pure object-oriented language is one in which every value is an object.

*Q :* Is Scala a pure OO language ? 

At first primitive types and functions seem like exceptions.   

**Primitive types** are in fact not implemented as Class. (E.g `scala.Int`: 32bits , `scala.Boolean` as Java Boolean ) , for reasons of efficiency. 

But they are not *conceptually* treated differently. In fact one can implement them using classes : 

```scala
package idealized.scala
abstract class Boolean extends AnyVal:
    def ifThenElse[T](t: => T, e: => T): T
    def && (x: => Boolean): Boolean = ifThenElse(x, false)
    def || (x: => Boolean): Boolean = ifThenElse(true, x)
    def unary_!: Boolean = ifThenElse(false, true)
    def == (x: Boolean): Boolean = ifThenElse(x, x.unary_!)
    def != (x: Boolean): Boolean = ifThenElse(x.unary_!, x)
...
end Boolean
object true extends Boolean:
def ifThenElse[T](t: => T, e: => T) = t
object false extends Boolean:
def ifThenElse[T](t: => T, e: => T) = e
```

   [Here](https://tube.switch.ch/videos/9081e1da) is how to implement `Scala.Int`. 

**Functions : **In scala functions are `objects` with `apply` methods

The function type `A => B` is just an abbreviation for the class
`scala.Function1[A, B]`, which is defined as follows.

```scala 
package scala
trait Function1[A, B]:
	def apply(x: A): B
```

Example : 

```scala
f = (x:Int) => x * x
// is expended to 
f = new Function1[Int, Int]:
	def apply(x: Int) = x * x
```

This anonymous class can itself expend to 

```scala
{ class $anonfun() extends Function1[Int, Int]:
def apply(x: Int) = x * x
$anonfun()
}
f(7) // expends to
f.apply(7)
```



