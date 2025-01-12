# 
# # NumPy

# 
# Read the links: https://numpy.org/doc/stable/user/quickstart.html  and https://numpy.org/doc/stable/user/basics.broadcasting.html  before solving the exercises. 

# %%
import numpy as np

# 
# ### Print out the dimension (number of axes), shape, size and the datatype of the matrix A.

# %%
A = np.arange(1, 16).reshape(3,5)

# %%
print(A.ndim)
print(A.shape)
print(A.size)
print(A.dtype)

# 
# ### Do the following computations on the matrices B and C: 
# * Elementwise subtraction. 
# * Elementwise multiplication. 
# * Matrix multiplication (by default you should use the @ operator).

# %%
B = np.arange(1, 10).reshape(3, 3)
C = np.ones((3, 3))*2

print(B)
print()
print(C)

# %%
print(B - C)
print(B * C)
print(B @ C)

# 
# ### Do the following calculations on the matrix:
# * Exponentiate each number elementwise (use the np.exp function).
# 
# * Calculate the minimum value in the whole matrix. 
# * Calculcate the minimum value in each row. 
# * Calculcate the minimum value in each column. 
# 
# 
# * Find the index value for the minimum value in the whole matrix (hint: use np.argmin).
# * Find the index value for the minimum value in each row (hint: use np.argmin).
# 
# 
# * Calculate the sum for all elements.
# * Calculate the mean for each column. 
# * Calculate the median for each column. 

# %%
B = np.arange(1, 10).reshape(3, 3)
print(B)

# %%
print(np.exp(B))
print(np.min(B))
print(np.min(B, axis=1))
print(np.min(B, axis=0))
print(np.argmin(B))
print(np.argmin(B, axis=1))
print(np.sum(B))
print(np.mean(B, axis=0))
print(np.median(B, axis=0))

# 
# ### What does it mean when you provide fewer indices than axes when slicing? See example below.

# %%
print(A)

# %%
A[1]

# 
# **Answer:**

# %%
### When you provide fewer indices than we have axes numpy assume that we want all elements along the missing axes. A1 means that it gives me the second row (index 1).

#
# ### Iterating over multidimensional arrays is done with respect to the first axis, so in the example below we iterate trough the rows. If you would like to iterate through the array *elementwise*, how would you do that?

# %%
A

# %%
for i in A:
    print(i)

# %%
for row in A:
    for element in row:
        print(element)

# %%
for element in np.nditer(A):
    print(element)

# %%
### You can use nested loops and np.nditer() that is more efficient and handles more complex cases.
### np.nditer is a efficient multi-dimensional iterator object to iterate over arrays (numpy.org). 

# 
# ### Explain what the code below does. More specifically, b has three axes - what does this mean? 

# %%
a = np.arange(30)
b = a.reshape((2, 3, -1))
print(a)
print()

print(b)

# %%
### b is an 3 dimensional array. 2 stands for 2 blocks of data, 3 stands for that each block of datavhas 3 rows and 5 is that each row has 5 elements. -1 turns to 5 because Numpy automatically 
### determine the size of the third axis so Numpy sets the size of the third axis to 5 to make the total number of the elements equal to 30. -1 is a way to tel Numpy that it can calculate the axis
### for me instead from manually figuring out the size of that axis by yourself. 

# 
# # For the exercises below, read the document *"matematik_yh_antonio_vektorer_matriser_utdrag"*
# # Solutions to the exercises and recorded videos can be found here: https://github.com/AntonioPrgomet/matematik_foer_yh
# 
# # If you find the exercises below very hard, do not worry. Try your best, that will be enough. 

# 
# ### Broadcasting
# **Read the following link about broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html#basics-broadcasting**

# 
# # Remark on Broadcasting when doing Linear Algebra calculations in Python. 

# 
# ### From the mathematical rules of matrix addition, the operation below (m1 + m2) does not make sense. The reason is that matrix addition requires two matrices of the same size. In Python however, it works due to broadcasting rules in NumPy. So you must be careful when doing Linear Algebra calculations in Python since they do not follow the "mathematical rules". This can however easily be handled by doing some simple programming, for example validating that two matrices have the same shape is easy if you for instance want to add two matrices. 

# %%
m1 = np.array([[1, 2], [3, 4]])
m2 = np.array([1, 1])
print(m1 + m2)

# 
# ### The example below would also not be allowed if following the "mathematical rules" in Linear Algebra. But it works due to broadcasting in NumPy. 

# %%
v1 = np.array([1, 2, 3])
print(v1 + 1)

# %%
A = np.arange(1, 5).reshape(2,2)
print(A)

b = np.array([2, 2])
print(b)

# 
# # Vector- and matrix algebra Exercises

# 
# **Now you are going to create a function that can be reused every time you add or multiply matrices. The function is created so that we do the addition and multiplication according to the rules of vector- and matrix algebra.**
# 
# **Create a function "add_mult_matrices" that takes two matrices as input arguments (validate that the input are of the type numpy.ndarray by using the isinstance function), a third argument that is either 'add' or 'multiply' that specifies if you want to add or multiply the matrices (validate that the third argument is either 'add' or 'multiply'). When doing matrix addition, validate that the matrices have the same size. When doing matrix multiplication, validate that the sizes conform (i.e. number of columns in the first matrix is equal to the number of rows in the second matrix).**

# %%
def add_mult_matrices(arg1,arg2,operation):
    if not isinstance(arg1, np.ndarray) or not isinstance (arg2, np.ndarray):
        raise ValueError ('the given arguments are not matrices')

    if operation == 'add':
        if arg1.size == arg2.size:
            return arg1 + arg2
        else:
            raise ValueError ('The matrices do not have the same sizze..')
    elif operation == 'multiply':
        if arg1.size == arg2.size:
            return arg1 * arg2
        else: 
            raise ValueError ('The matrices do not have the same shape..')
A = np.array ([[1, 2, 3]])
B = np.array ([[9, 8, 7]])
print(add_mult_matrices(A, B, 'add'))
print(add_mult_matrices(A, B, 'multiply'))

# 
# ### Solve all the exercises in chapter 10.1 in the book "Matematik för yrkeshögskolan" by using Python. 

# %%
# Uppgift 10.1.1. Definiera vektorn X enligt nedan.

x = np.array([4, 3])

#a: Vilken dimension har vektorn x?
#svar: Dimensionen x har två komponenter så vektorn x är 2 då den består av 2 tal.
print("a")
print(x.shape)

#b: Beräkna 5x
#svar: Då x=5 multiplicerar vi 5x4=20 och 5x3=15, svar blir 5x=(20, 15).
print("b")
five_x = 5 *  x
print(five_x)

# c: Beräkna 3x
# svar: Då x=3 multiplicerar vi 3x4=12 och 3x3=9, svar blir 3x=(12, 9). 
print("c")
three_x = 3 * x
print(three_x)

#d: Beräkna 5x + 3x
#svar: Från tidigare uppgift fick vi fram att 5x=(20,15) och 3x=(12, 9). Nu adderar vi dessa resultat och får (20, 15)+(12, 9)=(20+12)+(15+9=24) och får svaret (32, 24).
print("d")
result_add = five_x + three_x
print(five_x + three_x)

#e: Beräkna 8x
#svar: Då x=8 multiplicerar vi 8x4=32 och 8x3=24, svar blir 8x=(32, 24).
print("e")
eight_x = 8 * x
print(eight_x)

#f: Beräkna 4x-x
#svar: Då x=4 blir subtraktionen 4x4 + 4x3 - x=(4 ,3) = (16, 12) - (4, 3) = (12, 9)
print("f")
four_x = 4 * x
print(four_x - x)

#g: Beräkna xT, vilken blir den nya dimensionen efter att transponeringen utförts?
#svar: Radvektorn omvandlats till en kolumnvektor. 
print("g")
x_transponering = x.reshape(-1, 1)
print(x_transponering)

#h: Är x + xT definierat?
#svar: Nej, då vektorerna inte har samma dimensioner. 
print("h")
def_xT = x + x_transponering
print(def_xT)

#i= Beräkna ||x||.
#svar: ||x|| = roten av 4^2 + 3^2 = 25, roten av 25≈5
print("i")
normen_x = np.linalg.norm(x)
print(normen_x)





# %%
# Uppgift 10.1.2 Definiera vektorn v enligt nedan.

v = np.array([3, 7, 0, 11])

#a: Vilken dimension har vektorn v?
#Svar:
print("a")
print(v.shape)

#b: Beräkna 2v
#Svar:
print("b")
two_v = 2 * v
print(two_v)

#c: Beräkna 5v +  2v
#svar:
print("c")
five_v = 5 * v
result_add = five_v + two_v 
print(five_v + two_v)

#d: Beräkna 4v - 2v
#svar: 
print("d")
four_v = 4 * v 
result_add = four_v - two_v
print(four_v - two_v)

#e: Beräkna vT, vilken blir den nya dimensionen efter att transponering utförts?
#svar: 
print("e")
v_transponering = v.reshape(1, -1)
print(v_transponering)

#f: beräkna ||v||
#svar: 
print("f")
normen_v = np.linalg.norm(v)
print(normen_v)



# %%
#Uppgift 10.1.3 Definiera vektorerna v1 = (4, 3, 1, 5) och v2= (2, 3, 1, 1)

v1 = np.array ([4, 3, 1, 5])
v2 = np.array ([2, 3, 1, 1])

#a: Beräkna ||v1||.
#svar: 
print("a")
normen_v1 = np.linalg.norm(v1)
print(normen_v1)

#b: Beräkna ||v1-v2||
#Svar:
print("b")
normen_v2 = np.linalg.norm(v2)

result_add = normen_v1 - normen_v2
print(result_add)

# 
# ### Solve all the exercises, except 10.2.4, in chapter 10.2 in the book "Matematik för yrkeshögskolan" by using Python. 

# %%
#Uppgift 10.2.1. Definiera matriserna:

A = np.array([[2, 1, -1], [1, -1, 1]])
B = np.array([[4, -2, 1], [2, -4, -2]])
C = np.array([[1, 2], [2, 1]])
D = np.array([[3, 4], [4, 3]])
E = np.array([1, 2])
I = np.array([[1, 0], [0, 1]])

#a: Beräkna 2A
print("a")
two_A = 2 * A
print(two_A)

#b: Beräkna B-2A
#svar: 
print("b")
result = B - two_A
print(result)

#c: Beräkna 3C-2E
#svar: (egentligen ej definierat)
print("c")
three_C = 3 * C
two_E = 2 * E
result = three_C - two_E
print(result)

#d: Beräkna 2D - 3C
#svar: 
print("d")
two_D = 2 * D
three_C = 3 * C
result = two_D - three_C
print(result)

#e: Beräkna DT + 2D
#svar: 
print("e")
D_transponering = D.T
result = D_transponering + two_D
print(result)

#f: Beräkna 2CT - 2DT
#svar: 
print("f")
two_C = 2 * C
two_c = np.array
two_C_transponering = two_C.T
two_D_transponering = two_D.T
result = two_C_transponering - two_D_transponering
print(result)

#g: Beräkna AT-B
#svar: 
print("g")
A_transponering = A.T
try:
    result = A_transponering - B
except:
    print("Ej definierat")

#h: Beräkna AC
#Svar:
print("h")
try:
    result = A @ C
    print(result)
except:
    print("Ej definierat")


# %%
#i: Beräkna CD
#Svar: 
print("i")
result = C @ D
print(result)

#j: Beräkna CB
#Svar: 
print("j")
result = C @ B
print(result)

#k: Beräkna CI
#Svar: 
print("k")
result = C @ I
print(result)

#l: Beräkna ABT
#svar:
print("l")
B_transponering = B.T
result = A @ B_transponering
print(result)

#print(A @ B.T)

# %%
#Uppgift 10.2.2 Definiera matrisen
A = np.array([[2, 3, 4], [5, 4, 1]])

A_transponering = A.T
result = A @ A_transponering
print(result)

# %%
#Uppgift 10.2.3. Definiera matriserna

A = np.array([[1, 2], [2, 4]])
B = np.array([[2, 1], [1, 3]])
C = np.array([[4, 3], [0, 2]])

AB = A @ B
AC = A @ C
BC = B @ C


if np.array_equal(AB, AC):
    print("AB är lika med AC")
else:
    print("AB är inte lika med AC")
    
print(A @ B)
print(A @ C)

if np.array_equal(B, C):
    print("B är lika med C")
else:
    print("B är inte lika med C")

print(B @ C)

# 
# ### Copies and Views
# Read the following link: https://numpy.org/doc/stable/user/basics.copies.html

# 
# **Basic indexing creates a view, How can you check if v1 and v2 is a view or copy? If you change the last element in v2 to 123, will the last element in v1 be changed? Why?**

# %%
v1 = np.arange(4)
v2 = v1[-1:]
print(v1)
print(v2)

# %%
# The base attribute of a view returns the original array while it returns None for a copy.
print(v1.base)
print(v2.base)

# %%
# The last element in v1 will be changed aswell since v2 is a view, meaning they share the same data buffer.
v2[-1] = 123
print(v1)
print(v2)

# %%
# Answer: v1 is a copy, v2 is original array, you can check by use the base attribute. Any changes made to a view reflects the original copy. However changes to the copy do not reflect on the original array. 

# %%
#If you change the last element in v2 to 123, will the last element in v1 be changed? Why?
#Answer: Yes because v2 is a view into v1 and we can see that the last element of v1 changes to 123. 


