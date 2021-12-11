from PIL import Image
import os
import math
import numpy as np
import sys


key_words = ['png', 'xyz', 'color', 'trif', 'loadmv', 'loadp', 'xyzw', 'frustum', 'ortho', 'translate', 'rotatex', 'rotatey', 'rotatez', 'scale', 'multmv', 'trig', 'cull', 'lookat']
key_words.extend(['sphere', 'sun', 'bulb', 'plane', 'expose'])


def set_color(input_of_rgb):
    res = []
    for c in input_of_rgb:
        c = 12.92 * c if c <= 0.0031308 else 1.055 * (c**(1/2.4)) - 0.055
        if c <=0:
            res.append(0)
        elif c >= 1:
            res.append(255)
        else:
            res.append(int(c*255))
    return tuple(res)



def compObj(first, second):
    if first[0] == second[0]:
        if first[0] == 'sphere' and first[2] == second[2]:
            return np.all(first[1]) == np.all(second[1])
        elif first[0] == 'plane':
            return np.all([first[1:5]]) == np.all(second[1:5])
    return False



def raySphereIntersect(ro, rd, c, r):
    inside = np.linalg.norm(c - ro)**2 < r**2
    tc = np.dot((c - ro), rd)/np.linalg.norm(rd)
    if not inside and tc < 0:
        return np.inf
    d_sq = np.linalg.norm(ro + tc * rd - c)**2
    if not inside and d_sq > r**2:
        return np.inf
    t_offset = np.sqrt(r**2 - d_sq)/np.linalg.norm(rd)
    if inside:
        return tc+t_offset
    else:
        return tc-t_offset


def rayPlaneIntercept(ro, rd, a, b, c, d):
    normal = np.array([a,b,c])
    if a: 
        p = [-d/a,0,0]
    elif b: 
        p = [0,-d/b,0]
    else: 
        p = [0,0,-d/c]
    if np.dot(rd, normal) == 0: 
        return np.inf
    t = np.dot((p - ro), normal) / np.dot(rd, normal)
    return t



def illuminateSphere(ro, rd, t, c, r, obj_color, suns, bulbs, obstacles, exp):
    color = np.array([0,0,0])
    p = ro + t * rd
    normal = (p-c) / r
    if np.dot(normal,rd)>0:
        normal *= -1

    for sun in suns:
        sun_dir = np.array(sun[:3])
        sun_color = np.array(sun[3:])
        sun_dir = sun_dir/np.linalg.norm(sun_dir)
        
        dark = False
        for obj in obstacles:
            if obj[0] == 'sphere':
                calc = raySphereIntersect(p, sun_dir, obj[1], obj[2])
                dark = calc != np.inf

            if obj[0] == 'plane':
                calc = rayPlaneIntercept(p, sun_dir, obj[1], obj[2], obj[3], obj[4])
                dark =  calc != np.inf and calc >0
            if dark: 
                break

        if np.dot(normal,sun_dir)<0 or dark:
            continue
        color = color + obj_color * sun_color * np.dot(normal, sun_dir)
   
    for bulb in bulbs:
        bulb_dir = np.array(bulb[:3])-p
        bulb_color = np.array(bulb[3:])
        d = np.linalg.norm(bulb_dir)
        bulb_dir = bulb_dir / np.linalg.norm(bulb_dir)

        dark = False
        for obj in obstacles:
            if obj[0] == 'sphere':
                calc = raySphereIntersect(p, bulb_dir, obj[1], obj[2])
                dark = calc != np.inf and calc > 0 and np.linalg.norm(calc * bulb_dir) < d

            if obj[0] == 'plane':
                calc = rayPlaneIntercept(p, bulb_dir, obj[1], obj[2], obj[3], obj[4])
                dark = calc != np.inf and calc > 0 and np.linalg.norm(calc * bulb_dir) < d
            if dark: 
                break

        if np.dot(normal,bulb_dir)<0 or dark:
            continue
        color = color + obj_color * bulb_color * np.dot(normal, bulb_dir) / (d**2)

    if exp:
        color = np.subtract(1, np.exp(np.multiply(color, -exp)))

    color = set_color(color)
    return color



def illuminatePlane(ro, rd, t, a, b, c, d, obj_color, suns, bulbs, obstacles, exp):
    color = np.array([0,0,0])
    p = ro + t * rd
    normal = np.array([a,b,c])
    normal = normal / np.linalg.norm(normal)

    for sun in suns:
        sun_dir, sun_color = np.array(sun[:3]), np.array(sun[3:])
        sun_dir = sun_dir/np.linalg.norm(sun_dir)

        dark = False
        for obj in obstacles:
            if obj[0] == 'sphere':
                calc = raySphereIntersect(p, sun_dir, obj[1], obj[2])
                dark = calc != np.inf

            if obj[0] == 'plane':
                calc = rayPlaneIntercept(p, sun_dir, obj[1], obj[2], obj[3], obj[4])
                dark =  calc != np.inf and calc > 0
            if dark: 
                break

        if np.dot(normal,sun_dir)<0 or dark:
            continue
        color = color + obj_color*sun_color*np.dot(normal, sun_dir)


    for bulb in bulbs:
        bulb_dir, bulb_color = np.array(bulb[:3])-p, np.array(bulb[3:])
        d = np.linalg.norm(bulb_dir)
        bulb_dir = bulb_dir / d

        dark = False
        for obj in obstacles:
            if obj[0] == 'sphere':
                calc = raySphereIntersect(p, bulb_dir, obj[1], obj[2])
                dark = calc != np.inf and calc > 0 and np.linalg.norm(calc * bulb_dir) < d

            if obj[0] == 'plane':
                calc = rayPlaneIntercept(p, bulb_dir, obj[1], obj[2], obj[3], obj[4])
                dark = calc != np.inf and calc > 0 and np.linalg.norm(calc * bulb_dir) < d
            if dark: 
                break

        if np.dot(normal, bulb_dir) < 0 or dark:
            continue
        color = color + obj_color * bulb_color * np.dot(normal, bulb_dir) / (d**2)
    if exp:
        color = np.subtract(1, np.exp(np.multiply(color, -exp)))

    color = set_color(color)

    return color



def draw(image, width, height, objs, suns, bulbs, exp):
    eye = np.array([0,0,0])
    forward = np.array([0,0,-1])
    right = np.array([1,0,0])
    up = np.array([0,1,0])
    max_w_h = max(width, height)
    collection = []

    for obj in objs:
        if obj[0] == 'sphere':
            op, x, y, z, r, red, g, b = [thing for thing in obj]
            c = np.array([x, y, z])
            collection.append(['sphere',c, r, (red, g, b)])

        elif obj[0] == 'plane':
            op, a, b, c, d, red, g, b = [thing for thing in obj]
            collection.append(['plane', a, b, c, d, (red, g, b)])


    for x in range(width):
        for y in range(height):
            collision = []
            sx, sy = (2 * x - width) / max_w_h, (height - 2 * y) / max_w_h
            ro = eye
            ray = forward + sx * right + sy * up
            rd = ray / np.linalg.norm(ray)    

            for obj in objs:
                if obj[0] == 'sphere':
                    c = np.array(obj[1:4])
                    r = obj[4]
                    obj_color = np.array(obj[5:])
                    t = raySphereIntersect(ro, rd, c, r)
                    if t != np.inf and t > 0:
                        collision.append(['sphere',t, c, r, obj_color])
                    
                elif obj[0] == 'plane':
                    a, b, c, d = obj[1:5]
                    obj_color = np.array(obj[5:])
                    t = rayPlaneIntercept(ro, rd, a, b, c, d)
                    if t != np.inf and t > 0:
                        collision.append(['plane', t, a, b, c, d, obj_color])

            if collision:
                final_obj = sorted(collision,key=lambda cols: cols[1])[0]
                obj_infront = final_obj[:]
                obj_infront.pop(1)
                #obs_noself: keep only the spheres at the back
                obs_noself = []
                for ele in collection:
                    if not compObj(ele,obj_infront):
                        obs_noself.append(ele)

                if final_obj[0] == 'sphere':
                    t, c, r, obj_color = final_obj[1:]
                    color = illuminateSphere(ro, rd, t, c, r, obj_color, suns, bulbs, obs_noself, exp)

    
                elif final_obj[0] == 'plane':
                    t, a, b, c, d, obj_color = final_obj[1:]
                    color = illuminatePlane(ro, rd, t, a, b, c, d, obj_color, suns, bulbs, obs_noself, exp)

                image.im.putpixel((x,y), tuple(color))

    return image





def run(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines if line.rstrip() != '']
        lines = [line for line in lines if line.split()[0] in key_words]

    mega_info = lines[0].split()
    width = int(mega_info[1])
    height = int(mega_info[2])

    objs = []
    suns = [] 
    bulbs = []
    exp = 0

    if mega_info[0] == 'png':

        filename = mega_info[3]
        color_cur = (1,1,1)
        image = Image.new("RGBA", (width, height), (0,0,0,0))

        for line in lines[1:]:
            info = line.split()
            op = info[0]

            if op == 'color':
                color_cur = (float(info[1]),float(info[2]),float(info[3]))

            elif op == 'sphere':
                x, y, z, r = float(info[1]),float(info[2]), float(info[3]), float(info[4])
                objs.append([op, x, y, z, r, *color_cur])

            elif op == 'sun':
                x, y, z = float(info[1]),float(info[2]), float(info[3])
                suns.append([x, y, z, *color_cur])

            elif op == 'bulb':
                x, y, z = float(info[1]),float(info[2]), float(info[3])
                bulbs.append([x, y, z, *color_cur])

            elif op == 'plane':
                x, y, z, r = float(info[1]),float(info[2]), float(info[3]), float(info[4])
                objs.append([op, x, y, z, r, *color_cur])

            elif op == 'expose':
                exp = float(info[1])

        image = draw(image, width, height, objs, suns, bulbs, exp)
        return image, filename



if __name__ == "__main__":
    image, filename = run(sys.argv[1])
    image.save(filename)

