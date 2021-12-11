from PIL import Image
import os

key_words = ['xy', 'xyrgb', 'xyc', 'png', 'pngs']
with open ('implemented.txt') as files_needed:
    files_all = files_needed.readlines()
    files_all = [f.rstrip() for f in files_all]

# for filename in os.listdir('./'):
    # if filename.endswith('.txt'):
    for filename in files_all:

        with open(filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            lines = [line for line in lines if line.split()[0] in key_words]

        mega_info = lines[0].split()
        width = int(mega_info[1])
        height = int(mega_info[2])

        if mega_info[0] == 'png':

            filename = mega_info[3]

            image = Image.new("RGBA", (width, height), (0,0,0,0))
            for line in lines[1:]:
                info = line.split()
                op = info[0]
                if op == 'xy':
                    image.im.putpixel((int(info[1]),int(info[2])), (255, 255, 255, 255))
                elif op == 'xyrgb':
                    image.im.putpixel((int(info[1]),int(info[2])), (int(info[3]), int(info[4]), int(info[5]), 255))
                elif op == 'xyc':
                    color = info[3][1:]
                    image.im.putpixel((int(info[1]),int(info[2])), (int(color[0:2], 16),int(color[2:4], 16),int(color[4:6], 16), 255))
            
            image.save(filename)

        elif mega_info[0] == 'pngs':
            prefix = mega_info[3]
            count = int(mega_info[4])
            print(count)
            i = 0

            for line in lines[1:]:
                image = Image.new("RGBA", (width, height), (0,0,0,0))
                filename = str(prefix) + str(str(i).zfill(3)) + '.png'

                info = line.split()
                op = info[0]
                if op == 'xy':
                    image.im.putpixel((int(info[1]),int(info[2])), (255, 255, 255, 255))
                elif op == 'xyrgb':
                    image.im.putpixel((int(info[1]),int(info[2])), (int(info[3]), int(info[4]), int(info[5]), 255))
                elif op == 'xyc':
                    color = info[3][1:]
                    image.im.putpixel((int(info[1]),int(info[2])), (int(color[0:2], 16),int(color[2:4], 16),int(color[4:6], 16), 255))
                image.save(filename)
                i += 1
                    


# x = 2
# y = 3
# red = 0
# green = 134
# blue = 0
# alpha = 255
# filename = 'try.png'

# ...
# image.im.putpixel((x,y), (red, green, blue, alpha))
# ...