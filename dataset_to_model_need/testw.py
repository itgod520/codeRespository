import os
import re
"""批量修改文件夹的图片名"""
def ReFileName(dirPath,pattern):
    """
    :param dirPath: 文件夹路径
    :pattern:正则

    :return:
    """
    # 对目录下的文件进行遍历
    i = 1
    for file in os.listdir(dirPath):
        # 判断是否是文件

        if os.path.isfile(os.path.join(dirPath, file)) == True:
           #c= os.path.basename(file)
           print(file)
           newName = re.sub(pattern, "Test003_{:03d}.jpg".format(i), file)
           print("newName:",newName)
           newName=newName[:-15]
           newFilename = file.replace(file, newName)
           # 重命名
           os.rename(os.path.join(dirPath, file), os.path.join(dirPath, newFilename))
           i+=1
    print("图片名已全部修改成功")

if __name__ == '__main__':

    dirPath = r"D:\dataset\train\UMN3\test\Test003"
    pattern = re.compile(r'.*')
    ReFileName(dirPath,pattern)




