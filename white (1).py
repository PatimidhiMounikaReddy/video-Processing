import cv2
import numpy as np
import imutils
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd
import csv
from itertools import islice


# 1 = front, 2 = profile

# Create a VideoCapture object and read from input file
cap1 = cv2.VideoCapture('crop2.mp4')
cap2 = cv2.VideoCapture('profile.mp4')

coords1 = []
coords2 = []
coordsn2 = []
XL1 = []
YL1 = []
XL2 = []
YL2 = []     # z coordinates



while(1):

    # Take each frame
    _, frame1 = cap1.read()
    _, frame2 = cap2.read()

    # Blur the frame to reduce high frequency noise
    blurred1 = cv2.GaussianBlur(frame1, (11, 11), 0)
    blurred2 = cv2.GaussianBlur(frame2, (11, 11), 0)

    # Convert BGR to HSV
    hsv1 = cv2.cvtColor(blurred1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(blurred2, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,90,90])
    upper_blue = np.array([130,255,255])

    # define range of yellow color in HSV
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([30,255,255])

    # define range of red color in HSV
    lower_red = np.array([170,115,115])
    upper_red = np.array([179,255,255])

    # define range of white color in HSV
    lower_white = np.array([0,0,252])
    upper_white = np.array([180,255,255])

    lower_white2 = np.array([0,0,230])
    upper_white2 = np.array([180,255,255])

    # Threshold the HSV image to get only blue colors
    #mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Threshold the HSV image to get only yellow colors
    mask1 = cv2.inRange(hsv1, lower_white, upper_white)
    mask2 = cv2.inRange(hsv2, lower_white2, upper_white2)

    # Threshold the HSV image to get only red colors
    #mask = cv2.inRange(hsv, lower_red, upper_red)

    # Threshold the HSV image to get only white colors
    #mask = cv2.inRange(hsv, lower_white, upper_white)


    # Merge
    #mask = cv2.bitwise_or(mask1, mask2)

    # A series of dilations and erosions to remove any small noise
    mask1 = cv2.erode(mask1, None, iterations=2)
    mask1 = cv2.dilate(mask1, None, iterations=2)

    mask2 = cv2.erode(mask2, None, iterations=2)
    mask2 = cv2.dilate(mask2, None, iterations=2)

    # Bitwise-AND mask and original image
    res1 = cv2.bitwise_and(frame1,frame1, mask= mask1)

    res2 = cv2.bitwise_and(frame2, frame2, mask=mask2)



    # Find contours (frontal)
    cnts1 = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts1 = imutils.grab_contours(cnts1)

    # loop over the contours
    for c in cnts1:
        # compute the center of the contour
        # The function cv2.moments() gives a dictionary of all moment values calculated
        # Image moment is a particular weighted average of image pixel intensities.
        M1 = cv2.moments(c)


        cX1 = int(M1["m10"] / M1["m00"])
        cY1 = int(M1["m01"] / M1["m00"])
        coord1 = cX1, cY1
        center1 = (cX1, cY1)

        # Save the coordinates every frame on the list
        coords1.extend(coord1)

        XL1 = np.array(coords1[0::2])
        YL1 = np.array(coords1[1::2])

        XR1 = np.array(coords1[2::4])
        YR1 = np.array(coords1[3::4])


        # draw the contour and center of the shape on the image
        cv2.drawContours(res1, [c], -1, (0, 255, 0), 2)
        cv2.circle(res1, center1, 2, (0,0,0), -1)
        #cv2.circle(res, (cX, cY), 1, (255, 255, 255), -1)


        # Find contours (profile)
    cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts2 = imutils.grab_contours(cnts2)

    # loop over the contours
    for c in cnts2:
        # compute the center of the contour
        # The function cv2.moments() gives a dictionary of all moment values calculated
        # Image moment is a particular weighted average of image pixel intensities.
        M2 = cv2.moments(c)

        cX2 = int(M2["m10"] / M2["m00"])
        cY2 = int(M2["m01"] / M2["m00"])
        coord2 = cX2, cY2
        center2 = (cX2, cY2)

        # Save the coordinates every frame on the list
        coords2.extend(coord2)

        coordsn2 = np.array(coords2)

        XL2 = np.array(coords2[0::2])
        YL2 = np.array(coords2[1::2])


        XR2 = np.array(coords2[2::4])
        YR2 = np.array(coords2[3::4])

        # draw the contour and center of the shape on the image
        cv2.drawContours(res2, [c], -1, (0, 255, 0), 2)
        cv2.circle(res2, center2, 2, (0, 0, 0), -1)
        # cv2.circle(res, (cX, cY), 1, (255, 255, 255), -1)

    print("Left x coordinates(frontal) : ", XL1)
    print()
    print("Left y coordinates(frontal) : ", YL1)
    print()
    print("Coordinates(frontal) : ", coords1)
    print()

    print("Left x coordinates(profile) : ", XL2)
    print()
    print("Left y coordinates(profile) : ", YL2)
    print()
    print("Coordinates(profile) : ", coords2)
    print()

    #cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
    #cv2.imshow('res',res)
    cv2.imshow('frontal', np.hstack([frame1, res1]))
    #cv2.imshow('profile', np.hstack([frame2, res2]))


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()

cv2.destroyAllWindows()


# Plot

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 13
        }


plt.figure(figsize=(10, 6))
plt.subplot(221)
plt.plot(XL1)
plt.title('"White Ball (x coordinates)"')
plt. xlabel('frame')
plt. ylabel('x coordinates')

plt.subplot(222)
plt.plot(-(YL1))
plt.title('"White Ball (-y coordinates)"')
plt. xlabel('frame')
plt. ylabel('-y coordinates')

plt.subplot(212)
plt.plot(XL1, -(YL1), 'bo', markersize=4)
plt.title('"White Ball (Frontal)"')
plt.text(195, -63, '. start point', fontdict=font)
plt. xlabel('x coordinates')
plt. ylabel('y coordinates')

plt.show()





# Save data into excel file


df1 = pd.DataFrame(XL1)
df1.to_csv('XL1.csv')

df2 = pd.DataFrame(YL1)
df2.to_csv('YL1.csv')

df3 = pd.DataFrame(XL2)
df3.to_csv('XL2.csv')


x = []
y = []
z = []

with open('XL1.csv','r') as csvfile:
    plots = csv.reader(csvfile)
    for row in islice(plots, 2, 1150):
        x.append(float(row[1]))

with open('XL2.csv','r') as csvfile:
    plots = csv.reader(csvfile)
    for row in islice(plots, 2, 1150):
        y.append(float(row[1]))

with open('YL1.csv','r') as csvfile:
    plots = csv.reader(csvfile)
    for row in islice(plots, 2, 1150):
        z.append(-float(row[1]))


print(x)
print(y)
print(z)

data = {'x' : x, 'y' : y, '-z' : z}
df4 = pd.DataFrame(data)
df4.to_csv('white_ball_coordinates.txt', index=False)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y, x, z, label = 'white ball')
ax.set_title('"White Ball"')
ax.text(168, 195, -63, 'start point')
ax.set_xlabel('$Y$')
ax.set_ylabel('$X$')
ax.set_zlabel('$-Z$')
plt.show()



# txt file
# reference
# hidden ball coordinate
