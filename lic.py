####!/usr/bin/env python3
####!/usr/bin/env python

# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
from sklearn.linear_model import LinearRegression
import sys
from matplotlib import ticker

nx = 1
nxe = None
ny = 2
nye = None
title = None
xlabel = "x"
ylabel1 = "y1"
ylabel2 = "y2"
bis = False

def welcome_msg():
    print("This programme generates a plot of three subplots:" )
    print(" 1) one column in function of the other")
    print(" 2) residuals of the linear regression fit")
    print(" 3) result of nx - ny substraction")
    print()                       
    print("Usage: python3 lic.py <filename> [<nx=>] [<nxe=>] [<ny=>] [<nye=>] [<title=>] [<xlabel=>] [<ylabel1=>] [<ylabel2=>] [<bisector=>]")
    print("nx -> number of column to use as x-axis, default nx = 1")
    print("nxe -> number of column to use as x-axis error, default nxe - None")
    print("ny -> number of column to use as y-axis, default ny = 3")
    print("nye -> number of column to use as y-axis error, default nye - None")
    print("title -> title of the plot, default title = None")
    print("xlabel -> title of x-axis, default xlabel = x")
    print("ylabel1 -> title of upper's plot y-axis, default ylabel1 = y1")
    print("ylabel2 -> title of lower's plot y-axis, default ylabel2 = y2")
    print("bis -> representation of 1:1 relation, True/False, default = False")

if len(sys.argv) < 2:
    welcome_msg()
    sys.exit(1)
try:
    filename = str(sys.argv[1])
    f = open(filename)
    f.close()
except FileNotFoundError:
    print("File does not exist, try again!")
    sys.exit(1)
for arg in sys.argv:
    if "nx=" in arg:
        try:
            nx = int(str(arg).replace("nx=",""))
        except ValueError:
            print("Wrong value of the argument! Integer needed")
    if "nxe=" in arg:
        try:
            nxe = int(str(arg).replace("nxe=",""))
        except ValueError:
            print("Wrong value of the argument! Integer needed")
    if "ny=" in arg:
        try:
            ny = int(str(arg).replace("ny=",""))
        except ValueError:
            print("Wrong value of the argument! Integer needed")
    if "nye=" in arg:
        try:
            nye = int(str(arg).replace("nye=",""))
        except ValueError:
            print("Wrong value of the argument! Integer needed")
    if "title=" in arg:
        try:
            title = str(arg).replace("title=","")
        except ValueError:
            print("Wrong value of the argument!")
    if "xlabel=" in arg:
        try:
            xlabel = str(arg).replace("xlabel=","")
        except ValueError:
            print("Wrong value of the argument!")
    if "ylabel1=" in arg:
        try:
            ylabel1 = str(arg).replace("ylabel1=","")
        except ValueError:
            print("Wrong value of the argument!")
    if "ylabel2=" in arg:
        try:
            ylabel2 = str(arg).replace("ylabel2=","")
        except ValueError:
            print("Wrong value of the argument!")
    if "bis=" in arg:
        try:
            bis = str(arg).replace("bis=","")
        except ValueError:
            print("Wrong value of the argument! Boolean needed")

#converting reg and bis variables from strings to boolean
def str2bool(v):
    return str(v).lower() in ("True", "true")
bis = str2bool(bis)

#loading data to pandas DataFrame
table = np.loadtxt(filename)
table = pd.DataFrame(table)

nx = nx - 1
if nxe is not None:
    nxe = nxe - 1
ny = ny - 1
if nye is not None:
    nye = nye - 1

#skipping zero rows 
index_nx = table[(table[nx] == 0)].index
table.drop(index_nx, inplace = True)
index_ny = table[(table[ny] == 0)].index
table.drop(index_ny, inplace = True)

x_data = table[nx]
y_data = table[ny]

#linear regression
length = len(table)
x = table[nx].values
y = table[ny].values
x = x.reshape(length, 1)
y = y.reshape(length, 1)

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)
model = LinearRegression().fit(x_data.reshape(-1,1),y_data)
y_pred = np.array(model.predict(x), dtype=np.float32)
y_resid = np.array(y_data - y_pred, dtype=np.float32)

print('intercept (a): ', model.intercept_)
print('coefficient (b): ', model.coef_) 

#calculating residuals for the middle plot 
table['residuals'] = y_resid
mu1 = table['residuals'].mean()
sigma1 = table['residuals'].std()

print('mu (residuals):  ', mu1)
print('sigma (residuals): ', sigma1)

#calculating ny - nx for the lower plot
table['substraction'] = table[ny]-table[nx]
mu2 = table['substraction'].mean()
sigma2 = table['substraction'].std()

print('mu (substraction):  ', mu2)
print('sigma (substraction): ', sigma2)

plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 
fig, (axes) = plt.subplots(nrows = 3, sharex = True, facecolor = 'lightgrey', edgecolor = 'black', frameon = True, num = None, figsize = (10, 30),gridspec_kw={
                           'height_ratios': [1.5, 0.8 ,1 ]})

plt.subplots_adjust(hspace = 0.1)
plt.xlabel(xlabel, fontsize = 23)
if title is not None:
    plt.suptitle(title, fontsize = 20, y = 0.94)
    plt.subplots_adjust(left=0.1, right=0.99, top=0.9, bottom=0.03)
    
### PM: upper chart
axes[0].grid()
axes[0].set_ylabel(ylabel1, fontsize = 23)
axes[0].set_aspect('auto')
if nxe is not None and nye is not None:
    axes[0].errorbar(table[nx],table[ny], xerr = table[nxe], yerr = table[nye], linestyle = "None", c = 'b', zorder = 1, label='', elinewidth = 0.5)
elif nxe is None and nye is not None:
    axes[0].errorbar(table[nx],table[ny], yerr = table[nye], linestyle = "None", c = 'b', zorder = 1, label='', elinewidth = 0.5)
elif nye is None and nye is not None:
    axes[0].errorbar(table[nx],table[ny], xerr = table[nxe], linestyle = "None", c = 'b', zorder = 1, label='', elinewidth = 0.5)
elif nye is None and nxe is None:
    axes[0].errorbar(table[nx],table[ny], linestyle = "None", c = 'b', zorder = 1, label='')
up_plot = axes[0].hexbin(x_data, y_data, gridsize = 200, bins = 1000, mincnt = 1, cmap = 'plasma')   

if bis == True:
    x_bis = np.linspace(table[nx].min()-0.1*table[nx].min(), table[nx].max()+0.1*table[nx].max(),10)
    y_bis = x_bis
    axes[0].plot(x_bis, y_bis, color = 'orange', label = "Bisector")
    axes[0].legend()
axes[0].plot(x_data, y_pred, color = 'red', label = "Linear Regression", linewidth = 1.75, linestyle = 'dashed')
axes[0].legend(fontsize = 20)

### PM: middle chart
axes[1].grid()
axes[1].set_aspect('auto')
middle_plot = axes[1].hexbin(x_data, y_resid, gridsize = 200, bins = 1000, cmap='plasma', mincnt = 1)
axes[1].hlines(mu1, table[nx].min(), table[nx].max(), color = 'red', label = '$\mu$')
axes[1].set_ylabel('Residuals', fontsize = 23)
axes[1].spines['top'].set_visible(False)
axes[1].hlines(mu1 + sigma1, table[nx].min(), table[nx].max(), linestyles = 'dashed', colors = 'red', label = '$\mu $ +/- $\sigma$')
axes[1].hlines(mu1 - sigma1, table[nx].min(), table[nx].max(), linestyles = 'dashed', colors = 'red')
axes[1].hlines(mu1 + 3*sigma1, table[nx].min(), table[nx].max(), linestyles = 'dashdot', colors = 'red', label = '$\mu$ +/- $3\sigma$')
axes[1].hlines(mu1 - 3*sigma1, table[nx].min(), table[nx].max(), linestyles = 'dashdot', colors = 'red')
axes[1].legend(fontsize = 20)#, loc = 'lower left')

### PM: lower chart
axes[2].grid()
axes[2].set_aspect('auto')
axes[2].hexbin(x_data, table['substraction'], gridsize = 200, bins = 1000, cmap = 'plasma', mincnt = 1)
axes[2].hlines(mu2, table[nx].min(), table[nx].max(), color = 'red', label = '$\mu$')
axes[2].set_ylabel(ylabel2, fontsize = 23)
axes[2].spines['top'].set_visible(False)
axes[2].hlines(mu2 + sigma2, table[nx].min(), table[nx].max(), linestyles = 'dashed', colors = 'red', label = '$\mu$ +/- $\sigma$')
axes[2].hlines(mu2 - sigma2, table[nx].min(), table[nx].max(), linestyles = 'dashed', colors = 'red')
axes[2].hlines(mu2 + 3*sigma2, table[nx].min(), table[nx].max(), linestyles = 'dashdot', colors = 'red', label = '$\mu$ +/- $3\sigma$')
axes[2].hlines(mu2 - 3*sigma2, table[nx].min(), table[nx].max(), linestyles = 'dashdot', colors = 'red')
axes[2].legend(fontsize = 20)#, loc = 'lower left')



cb = fig.colorbar(up_plot, orientation = 'horizontal')
tick_locator = ticker.MaxNLocator(nbins=13)
cb.locator = tick_locator
cb.update_ticks()

if title is None:
    plt.tight_layout()
    plt.subplots_adjust(left=0.16)

### PM: show and save
plt.savefig('plot.png', format = 'png')
plt.show()
print("Your plot has been saved under a name: plot.png") 
