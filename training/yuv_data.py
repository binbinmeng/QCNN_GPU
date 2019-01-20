# -*- coding: utf-8 -*-
"""
Created on Mar 9 15:50 2018

@author: Hengrui Zhao
"""

import numpy as np
from PIL import Image
import configparser
#'''
cfg=[#classA
    '../data/cfg/Traffic.ini',
    '../data/cfg/PeopleOnStreet.ini',
    #classB
    '../data/cfg/Kimono.ini',
    '../data/cfg/ParkScene.ini',
    '../data/cfg/Cactus.ini',
    '../data/cfg/BasketballDrive.ini',
    '../data/cfg/BQTerrace.ini',
    #classC
    '../data/cfg/BasketballDrill.ini',
    '../data/cfg/BQMall.ini',
    '../data/cfg/PartyScene.ini',
    '../data/cfg/RaceHorsesC.ini',
    #classD
    '../data/cfg/BasketballPass.ini',
    '../data/cfg/BQSquare.ini',
    '../data/cfg/BlowingBubbles.ini',
    '../data/cfg/RaceHorses.ini',
    #classE
    '../data/cfg/FourPeople.ini',
    '../data/cfg/Johnny.ini',
    '../data/cfg/KristenAndSara.ini']
#'''
#cfg=[ '../data/cfg/BlowingBubbles.ini']
frame=1
class YuvData:

    def __init__(self, q):
        self.testData=[]
        self.cfg = configparser.ConfigParser()
        self.QP=q
        for i in range(len(cfg)):
            self.cfg.read(cfg[i])
            self.testData.append([cfg[i]])
            self.testData[i].append(
                self.get_channel(
                    self.cfg['path']['ori'],
                    self.cfg.getint('size', 'height'),
                    self.cfg.getint('size', 'width'),
                    frame,#self.cfg.getint('size', 'frame'),
                    'Y'))
            self.testData[i].append(
                self.get_channel(
                    self.cfg['path']['file1'] + 'Q%d.yuv' % q,
                    self.cfg.getint('size', 'height1'),
                    self.cfg.getint('size', 'width1'),
                    frame,#self.cfg.getint('size', 'frame'),
                    'Y'))

    @staticmethod
    def get_channel(file, height, width, frame, channel):

        fp = open(file, 'rb')
        buf = fp.read()
        fp.close()

        y_size = height * width
        frame_size = y_size * 3 // 2

        if channel == 1 or 'Y':
            y = np.zeros((frame, height, width), np.uint8, 'C')
            for i in range(frame):
                y[i] = np.frombuffer(
                    buf[i * frame_size:i * frame_size + y_size], dtype=np.uint8).reshape(height, width)
            return y
        elif channel == 2 or 'U':
            u = np.zeros((frame, height // 2, width // 2), np.uint8, 'C')
            for i in range(frame):
                u[i] = np.frombuffer(buf[i * frame_size + y_size:i * frame_size + y_size + y_size // 4],
                                  dtype=np.uint8).reshape(height // 2, width // 2)
            return u
        elif channel == 3 or 'V':
            v = np.zeros((frame, height // 2, width // 2), np.uint8, 'C')
            for i in range(frame):
                v[i] = np.frombuffer(buf[i * frame_size + y_size + y_size // 4:i * frame_size + y_size + y_size // 2],
                                  dtype=np.uint8).reshape(height // 2, width // 2)
            return v
        else:
            print('No such channel in YUV format.')
            exit(1)

    def get_frame(self, seq=0, frame=0):
        if frame >= self.testData[seq][1].shape[0]:
            print("no such frame")
            exit(1)
        else:
            ori = self.testData[seq][1][frame]
            frame1 = self.testData[seq][2][frame]
            return [ori, frame1]

    def save_res(self):
        with open("res%d.data"%self.QP,"wb") as f:
            for item in self.testData:
                res=item[1].astype(np.int)-item[2].astype(np.int)
                f.write(res)

    def test(self):
        pass


if __name__ == '__main__':

    b = YuvData(22)
    b.save_res()
    #test_frame = b.get_frame(seq=3,frame=0)
    #im = Image.fromarray(np.concatenate((test_frame[0],test_frame[1]),axis=0))
    #im.show()
