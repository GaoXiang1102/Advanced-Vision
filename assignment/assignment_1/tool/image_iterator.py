import os

Const_Image_Format = [".jpg", ".jpeg", ".bmp", ".png"]


class FileFilt:
    fileList = [""]
    counter = 0

    def __init__(self):
        pass

    def FindFile(self, dirr, filtrate=1):
        global Const_Image_Format
        for s in os.listdir(dirr):
            newDir = os.path.join(dirr, s)
            if os.path.isfile(newDir):
                if filtrate:
                    if newDir and (os.path.splitext(newDir)[1] in Const_Image_Format):
                        self.fileList.append(newDir)
                        self.counter += 1
                else:
                    self.fileList.append(newDir)
                    self.counter += 1


if __name__ == "__main__":
    b = FileFilt()
    b.FindFile(dirr="../../../data/image/day_2")
    print(b.counter)
    for k in b.fileList:
        print k
