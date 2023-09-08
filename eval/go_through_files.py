import os 
import glob

files = r'D:\eigen_split'
for (dirpath, dirnames, filenames) in os.walk(files):
     for name in filenames:
            with open(r'C:\Users\ppapadop\Desktop\nyu2_eigen_test.csv', 'a') as f:
                a = os.path.join(dirpath,name)       
        
                f.write(a)
              

                #a,b,c = a.split('_')
                #f.write(a)
                #f.write('.png')
                #f.write(',')
                #f.write(name)
                f.write('\n')
       
  