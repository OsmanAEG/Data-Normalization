#Importing Numpy, Matplotlib, Math
import numpy as np
import matplotlib.pyplot as plt
import math

#Data Normalization
xi = np.array([13, 10, 12, 11, 12, 12, 13, 10, 16, 17, 16, 18, 17, 16, 19, 15, 17, 16, 19, 15, 18])
xi = np.sort(xi)

N = np.size(xi)
sum_xi = np.sum(xi)

mean = sum_xi/N
var = (1/(N-1))*np.sum((xi-mean)**2)
std_dev = math.sqrt(var)

xi_Star = (xi - mean)/std_dev

print('xi: ' + str(xi))
print('N: ' + str(N))
print('Sum xi: ' + str(sum_xi))
print('Mean: ' + str(mean))
print('Variance: ' + str(var))
print('Standard Deviation: ' + str(std_dev))
print('xi Star: ' + str(np.around(xi_Star, decimals = 1)))

num_bins = 5

x_min = -5.0
x_max = 5.0

n = 1000
dx = (x_max-x_min)/n

x = np.zeros(n)
func = np.zeros(n)

for i in range(n):
    x[i] = x_min + (i + 0.5)*dx
    func[i] = (1.0/math.sqrt((2.0*math.pi)))*math.exp(-(x[i])**2/(2.0))

#Chi-Squared Test
j1, n1, y1, p1 = np.zeros(0), 0, np.array([-1e12, -0.99]), 0.0
j2, n2, y2, p2 = np.zeros(0), 0, np.array([-0.99, 0.22]), 0.0
j3, n3, y3, p3 = np.zeros(0), 0, np.array([0.22, 0.74]), 0.0
j4, n4, y4, p4 = np.zeros(0), 0, np.array([0.74, 1e12]), 0.0

for i in range(n):
    if y1[0] < x[i] <= y1[1]:
        j1 = np.append(j1, func[i])
    elif y2[0] < x[i] <= y2[1]:
        j2 = np.append(j2, func[i])
    elif y3[0] < x[i] <= y3[1]:
        j3 = np.append(j3, func[i])
    elif y4[0] < x[i] <= y4[1]:
        j4 = np.append(j4, func[i])

for val in xi_Star:
    if y1[0] < val <= y1[1]:
        n1 += 1
    elif y2[0] < val <= y2[1]:
        n2 += 1
    elif y3[0] < val <= y3[1]:
        n3 += 1
    elif y4[0] < val <= y4[1]:
        n4 += 1

p1 = np.trapz(j1, dx=dx)
p2 = np.trapz(j2, dx=dx)
p3 = np.trapz(j3, dx=dx)
p4 = np.trapz(j4, dx=dx)

chi1 = (n1/N-p1)**2/p1
chi2 = (n2/N-p2)**2/p2
chi3 = (n3/N-p3)**2/p3
chi4 = (n4/N-p4)**2/p4

chi_square = chi1 + chi2 + chi3 + chi4
print('n1: ' +  str(n1) + ' | ' + 'y1: ' +  '-inf' + ' | ' + 'y2: ' +  str(y1[1]) + ' | ' + 'p1: ' +  str(p1))
print('n2: ' +  str(n2) + ' | ' + 'y1: ' +  str(y2[0]) + ' | ' + 'y2: ' +  str(y2[1]) + ' | ' + 'p2: ' +  str(p2))
print('n3: ' +  str(n3) + ' | ' + 'y1: ' +  str(y3[0]) + ' | ' + 'y2: ' +  str(y3[1]) + ' | ' + 'p3: ' +  str(p3))
print('n4: ' +  str(n4) + ' | ' + 'y1: ' +  str(y4[0]) + ' | ' + 'y2: ' +  '+inf' + ' | ' + 'p4: ' +  str(p4))
print('Chi Squared: ' + str(chi_square))

#Plotting
plt.hist(xi_Star, density=True, bins = num_bins)
plt.plot(x, func)
plt.title('Normalized Histogram')
plt.xlabel('Normalized Data')
plt.ylabel('Probability')
plt.show()

