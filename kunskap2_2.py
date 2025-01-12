# 
# # Matplotlib
# Read the tutorials: https://matplotlib.org/stable/users/explain/quick_start.html and https://matplotlib.org/stable/tutorials/pyplot.html before solving the exercises below. The "Pyplot Tutorial" you do not read in detail but it is good to know about since the fact that there are two approaches to plotting can be confusing if you are not aware of both of the approaches.

# 
import numpy as np
import matplotlib.pyplot as plt

# 
# ### Plotting in Matplotlib can be done in either of two ways, which ones? Which way is the recommended approach?

# 
### Explicity create Figures and Axes, and call methods on them (the "object-oriented (OO) style").
### Rely on pyplot to implicitly create and manage the Figures and Axes, and use pyplot functions for plotting (matplotlib.pyplot).

# 
# ### Explain shortly what a figure, axes, axis and an artist is in Matplotlib.

# 
### Figure = The figure keeps track of all the child Axes, a group of 'special' Artists (titles, figure legends, colorbars) and even nested subfigures. Typically, you'll create a new Figure through one of the following functions: subplots and subplot_mosaic. 
### Axes = An axes is an Artist attached to a Figure that contains a region for plotting data, and usually includes two (or three in the case of 3D) Axis objects. Each Axes also has a title (via set_title()) an X-lable and an Y-lable. The Axes methods are the primary interface for configuring most parts of your plot. 
### Axis = These objects set the scale and limits and generate ticks and ticklables. The location of the ticks is determined by a Locator object and the ticklabel strings are formatted by a Formatter. The combination of the correct Locator and Formatter gives very fine control over the tick locations and labels.
### Artist = Basically everything visible on the Figure is an Artist (even Figure, Axes and Axis objects). This includes Text objects, Line 2D objects, collections objects, Patch objects etc. When the Figure is rendered, all of the Artists are drawn to the canvas. Most Artist are tied to an Axes: such an Artist cannot be shared by multiple Axes or moved from one to another. 

# 
# ### When plotting in Matplotlib, what is the expected input data type?

# 
### numpy.array, numpy.asarray, Python tuples, Pandas DataFrame/Series, numpy.matrix, 2D arrays for images/maps.

# 
# ### Create a plot of the function y = x^2 [from -4 to 4, hint use the np.linspace function] both in the object-oriented approach and the pyplot approach. Your plot should have a title and axis-labels.

# 
###objectoriented approach

x = np.linspace(-4, 4 , 100)
y = x**2
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel('x lable')
ax.set_ylabel('y lable')
ax.set_title("Simple Plot")
plt.show

# 
###pyplot approach
x = np.linspace(-4, 4 ,100)
y = x**2
plt.plot(x, y)
plt.xlabel('x lable')
plt.ylabel('y lable')
plt.title("Simple Plot")
plt.show()

# 
# ### Create a figure containing 2  subplots where the first is a scatter plot and the second is a bar plot. You have the data below. 

# 
# Data for scatter plot
np.random.seed(15)
random_data_x = np.random.randn(1000)
random_data_y = np.random.randn(1000)
x = np.linspace(-2, 2, 100)
y = x**2

# Data for bar plot
fruit_data = {'grapes': 22, 'apple': 8, 'orange': 15, 'lemon': 20, 'lime': 25}
names = list(fruit_data.keys())
values = list(fruit_data.values())

# 
###Data for scatter plot

np.random.seed(15)
random_data_x = np.random.randn(1000)
random_data_y = np.random.randn(1000)
x = np.linspace(-2, 2, 100)
y = x**2
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(random_data_x, random_data_y, color='orange', alpha=0.7, label='Random Data')
ax.plot(x, y, color='green', label='y = x**2')
ax.set_title('Data for Random Scatter Plot')
ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
plt.show()

# 
###Data for bar plot

fruit_data = {'Grapes': 22, 'Apple': 8, 'Orange': 15, 'Lemon': 20, 'Lime': 25}
names = list(fruit_data.keys())
values = list(fruit_data.values())
fig, ax = plt.subplots(figsize=(10, 10))
ax.bar(names, values, color='green')
ax.set_title('Data for Fruits Bar Plot')
ax.set_xlabel('Fruits')
ax.set_ylabel('Count')
plt.show()


