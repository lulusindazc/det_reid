#---------------------------------
# @time             : 2018-11-03
# @Author           : Wison
# @Description      : Package the multi cams as one file
# @last_modification: 2018-11-03
#---------------------------------

import runcam1, runcam2, runcam3, runcam4
import rospy
import threading

def main():
    rospy.init_node('det_reid', anonymous=True)
    threadPub1 = threading.Thread(target=runcam1.main, args=[3])
    threadPub2 = threading.Thread(target=runcam2.main, args=[4])
    # threadPub1 = runcam1.main(3)
    threadPub1.setDaemon(True)

    # threadPub2 = runcam2.main(4)
    threadPub2.setDaemon(True)
    #
    # threadPub3 = runcam3.main(5)
    # threadPub3.setDaemon(True)
    #
    # threadPub4 = runcam4.main(6)
    # threadPub4.setDaemon(True)
    threadPub1.start()
    threadPub2.start()
    # threadPub3.start()
    # threadPub4.start()




    while not rospy.is_shutdown():
        pass

if __name__ == '__main__':
    main()