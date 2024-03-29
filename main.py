import data
import algorithm as al
import numpy as np
from sklearn.model_selection import train_test_split

x = data.x
x = (x - np.mean(x, axis = 0))/np.std(x, axis = 0)
t = data.t

x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.33, random_state=11152019)

model = al.algorithm()
model.fit(x_train, t_train)
print(model.w)
print(model.b)

y_predict = model.predict(x_test)

mse = np.dot( (t_test - y_predict)**2,  (t_test - y_predict)**2 )/len(t_test)
print("Mean square error = %.2f" %(mse))

tot = np.dot( t_test - np.mean(t_test), t_test - np.mean(t_test) )/len(t_test)
r_2 = 1 - ( mse / tot)
print("R square score = %.2f" %(r_2))

import matplotlib.pyplot as plt
plt.scatter(t_test, y_predict)
plt.plot(np.arange(-3,3,0.01), np.arange(-3,3,0.01), color = 'red')

plt.xlabel('t - true traget values') 
plt.ylabel('y - predicted values') 
plt.title('Comparision btw true and predict values') 

plt.xticks(())
plt.yticks(())

plt.savefig('result.png')