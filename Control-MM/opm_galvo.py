import serial
import time
import numpy as np

tgsCom    = "/dev/tty.usbmodem14101"  ##### MODIFY THIS LINE FOR YOUR SERIAL PORT NAME OR NUMBER
tgS = serial.Serial()
tgS.port = tgsCom
tgS.baudrate = 115200
tgS.bytesize = serial.EIGHTBITS  # number of bits per bytes
tgS.parity = serial.PARITY_NONE  # set parity check: no parity
tgS.stopbits = serial.STOPBITS_ONE  # number of stop bits
# tgS.timeout = None          #block read
tgS.timeout = 0.5  # non-block read
tgS.xonxoff = False  # disable software flow control
tgS.rtscts = False  # disable hardware (RTS/CTS) flow control
tgS.dsrdtr = False  # disable hardware (DSR/DTR) flow control
tgS.writeTimeout = 0  # timeout for write

try:
    print("Activating Triggerscope...")
    tgS.open()
except Exception as e:
    print("ERROR: Triggerscope Com port NOT OPEN: " + str(e))
    exit()
if tgS.isOpen():
    try:
        tgS.flushInput()  # flush input buffer, discarding all its contents
        tgS.flushOutput()  # flush output buffer, aborting current output
        op = "*"
        tgS.write(op.encode() + "\n".encode('ascii'))  # send an ack to tgs to make sure it's up
        time.sleep(0.2)  # give the serial port sometime to receive the data
        print("Rx: " + tgS.readline().decode())
    except Exception as e1:
        print(" serial communication error...: " + str(e1))

else:
    print("cannot open tg cell  port ")


def writetgs(tgin):
    '''send a serial command to the triggerscope...
    Args:
        tgin: input string to send. Note the command terminator should be included in the string.
    Returns:
        char string of whatever comes back on the serial line.
    Raises:
        none.
    '''
    tgS.flushInput()  # flush input buffer, discarding all its contents
    tgS.flushOutput()  # flush output buffer, aborting current output
    tgS.write(tgin.encode())  # send command
    time.sleep(0.02)  # give the serial port sometime to receive the data 50ms works well...
    bufa = ""
    bufa = tgS.readline()
    return bufa

def readStat():

    tgS.flushInput()  # flush input buffer, discarding all its contents
    tgS.flushOutput()  # flush output buffer, aborting current output
    tgS.write("STAT?\n".encode())  # send command
    time.sleep(0.02)  # give the serial port sometime to receive the data 50ms works well...
    for n in range(100):
        time.sleep(0.2)  # give the serial port sometime to receive the data 50ms works well...
        bufa = ""
        bufa = tgS.readline()
        print(bufa);
        if(len(bufa) < 5):
            break

def SpeedTestA():

   # for x in range (5):
   #     time.sleep(1)  # for the arduino reset
   #     print(x)
    start = time.time()

    print( writetgs("TIMECYCLES,3\n") )


    print( writetgs("PROG_TTL,1,1,1\n") )
    print( writetgs("PROG_TTL,1,2,0\n") )

    print( writetgs("PROG_TTL,2,1,0\n") )
    print( writetgs("PROG_TTL,2,2,1\n") )
    print( writetgs("PROG_TTL,2,3,0\n") )

    print( writetgs("PROG_TTL,3,1,0\n") )
    print( writetgs("PROG_TTL,3,2,0\n") )
    print( writetgs("PROG_TTL,3,3,1\n") )
    print( writetgs("PROG_TTL,3,4,0\n") )
    #readStat()
    input(" Press Enter to ARM...")

    print( writetgs("ARM\n") )

    for n in range(100):
        time.sleep(0.2)  # give the serial port sometime to receive the data 50ms works well...
        bufa = ""
        bufa = tgS.readline()
        print(bufa);
        if (len(bufa) < 5):
            break

def PWMGen():
    print(writetgs("TIMECYCLES,100\n")) #cycle this operation 100 times on camera input trigger
    print(writetgs("PROG_TTL,1,1,1\n"))
    print(writetgs("PROG_TTL,1,2,0\n"))
    print( writetgs("ARM\n") ) #Arm sequence
    for n in range(100): #read reply from triggerscope
        time.sleep(0.2)  # give the serial port sometime to receive the data 50ms works well...
        bufa = ""
        bufa = tgS.readline()
        print(bufa);
        if (len(bufa) < 5):
            break

def cyclettl():
    for i in range(30):
        print(writetgs("TTL3,1\n"))
        time.sleep(0.05)
        print(writetgs("TTL3,0\n"))
        time.sleep(0.05)

def cycledac():
    for i in range(30):
        print(writetgs("DAC1,"+str(2184*i)+"\n"))
        time.sleep(0.05)

def opmgalvo_v1():
    print( writetgs("ARM\n") ) #Arm sequence
    for i in range(100):
        if i % 2 > 0:
            print(writetgs("TTL3,0\n"))
        else:
            print(writetgs("TTL3,1\n"))
        print(writetgs("DAC1,"+str(655*i)+"\n"))
        time.sleep(0.05)

def opmgalvo_v2():
    # create_array()
    nframes = 20
    dac1 = np.zeros(nframes+1,dtype=int);
    ttl3 = np.zeros(nframes+1,dtype=int);
    j = 1;
    for i in range(nframes):
        if i % 2 == 0:
            dac1[i] = (65000/(0.5*nframes))*j
            ttl3[i] = 0;
        else:
            dac1[i] = dac1[i-1]
            ttl3[i] = 1;
        j = j + 0.5;    
    dac1[nframes] = 65000/2
    ttl3[nframes] = 0

    # send_serial
    for i in range(nframes+1):
        print(writetgs("DAC1,"+str(dac1[i])+"\n"))
        time.sleep(0.01)
        print(writetgs("TTL3,"+str(ttl3[i])+"\n"))
        time.sleep(0.01)

    # Arm_sequence
    print( writetgs("ARM\n") ) #Arm sequence

    for i in range(nframes+1):
        print(dac1[i],ttl3[i])

def opmgalvo_v3():
    # create_array
    nframes = 20
    dac = np.zeros((nframes+1,16),dtype=int);
    ttl = np.zeros((nframes+1,16),dtype=int);

    k = 1;
    for i in range(nframes):
        dac[i,1] = (65000/nframes)*(i+1)
        ttl[i,2] = 1;
        # if i % 2 == 0:
        #     dac[i,0] = (65000/(0.5*nframes))*k
        #     ttl[i,2] = 0
        # else:
        #     dac[i,0] = dac[i-1,0]
        #     ttl[i,2] = 1;
        # k = k + 0.5;    
    dac[nframes,1] = 65000/2
    ttl[nframes,2] = 0

    # send_serial
    print(writetgs("TRIGMODE,4\n"))
    print(writetgs("TIMECYCLES,100\n")) #cycle this operation 100 times on camera input trigger
    for i in range(nframes+1):
        for j in range(16):
            tout = "PROG_TTL," + str(i+1) + "," + str(j+1) + "," + str(ttl[i][j]) + "\n"
            print(writetgs(tout))
            time.sleep(0.01)
        for j in range(16):
            dout = "PROG_DAC," + str(i+1) + "," + str(j+1) + "," + str(dac[i][j]) + "\n"
            print(writetgs(dout))
            time.sleep(0.01)

    # Arm_sequence
    print( writetgs("ARM\n") ) #Arm sequence

    # print array
    dac_matrix = np.array(dac)
    ttl_matrix = np.array(ttl)
    print(dac_matrix)
    print(ttl_matrix)


opmgalvo_v3()

#input(" Check for TTL 1 on and off @ 10Hz - Arm Oscilloscope Press Enter ...")
#cyclettl()

# input(" Check for DAC 1 ramp @ 10Hz - Arm Oscilloscope Press Enter ...")
# cycledac()


#readStat()

# from pycromanager import Bridge
# 
# #Create the Micro-Managert to Pycro-Manager transfer layer
# bridge = Bridge()
# #get object representing micro-manager core
# core = bridge.get_core()
# 
# print(core)
# 
