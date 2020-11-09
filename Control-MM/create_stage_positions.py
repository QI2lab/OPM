#!/usr/bin/env python

import numpy as np

x_step=1280
z_step=150-(300)
viewID_offset = 223
counter=0

for z in range(3,0,-1):
    for x in range(0,37):
        for y in range(0,2):

            if z==3:
                x_step=1300

            x_val = x * x_step
            if y == 0:
                y_val = 0
            else:
                y_val = 21000-320+16
            z_val = (z-1)*z_step
            string_0 = str(counter+0*viewID_offset)+';;('+str(x_val)+','+str(y_val)+','+str(z_val)+')'
            string_1 = str(counter+1*viewID_offset)+';;('+str(x_val)+','+str(y_val)+','+str(z_val)+')'
            string_2 = str(counter+2*viewID_offset)+';;('+str(x_val)+','+str(y_val)+','+str(z_val)+')'
            string_3 = str(counter+3*viewID_offset)+';;('+str(x_val)+','+str(y_val)+','+str(z_val)+')'

            counter=counter+1

            print(string_0)
            print(string_1)
            print(string_2)
            print(string_3)