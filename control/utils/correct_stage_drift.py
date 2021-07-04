#!/usr/bin/env python
'''
Optimize primary stage XYZ position with respect to reference image during iterative run.

Process:
1. Setup automated iterative run.
2. Flag "optimize stage positions after first run" and select reference channel
3. First round (r=0) will run using stage scan only
4. Starting with second round (r>0), instrument will first run a 200 um galvo scan using reference channel
5. This galvo scan will be deskewed and compared to initial 200 um of stage scan from r=0
6. Calculate YZ offset between current and reference image using phase cross-correlation 
7. Offsets applied to YZ stage positions
8. Now, multicolor stage scan runs from offset positions
9. Repeats for every tile position in the dataset

Shepherd 07/21
'''

