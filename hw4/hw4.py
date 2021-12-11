from PIL import Image
import os
import math
import numpy as np
import sys
from numpy.linalg import inv


# key_words = ['xyrgb', 'xyc', 'png', 'pngs', 'trig', 'linec', 'lineg', 'tric', 'xyrgba', 'trica', 'polyec', 'polynz', 'fann', 'stripn', 'xyz', 'color', 'trif', 'loadmv', 'loadp', 'xyzw', 'frustum']
key_words = ['png', 'xyz', 'color', 'trif', 'loadmv', 'loadp', 'xyzw', 'frustum', 'ortho', 'translate', 'rotatex', 'rotatey', 'rotatez', 'scale', 'multmv', 'trig', 'cull', 'lookat']
key_words.extend(['pngs','object','position','quaternion','add','sub','mul','div','pow','sin','cos','origin','scale','anyscale','euler','iflt','else','fi','camera'])
dict_vertices = {}
def create_line_color(point1, point2, dict_vertices):
    points_to_fill = []
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    z1 = point1[2]
    z2 = point2[2]
    reverse = 0

    # assume x is longer; if not, switch x and y
    # assume x1 is smaller than y1
    #going from small x to large x
    if abs(x1-x2) < abs(y1-y2):
        reverse = 1
        y1 = point1[0]
        x1 = point1[1]
        y2 = point2[0]
        x2 = point2[1]

    if x1 > x2:
        x_temp = x2
        y_temp = y2
        x2 = x1
        y2 = y1
        x1 = x_temp
        y1 = y_temp
        z_temp = z2
        z2 = z1
        z1 = z_temp

    slope = (y2-y1)/(x2-x1)
    cur_x = math.ceil(x1)
    delta_x =cur_x - x1
    delta_y = delta_x * slope
    cur_y = y1 + delta_y

    slope_z = (z2-z1)/(x2-x1)
    delta_z = delta_x * slope_z
    cur_z = z1 + delta_z

    tot_points = math.floor(x2) - cur_x +1

    if not reverse:
        start_color = dict_vertices[(x1, y1, z1)]
        end_color = dict_vertices[(x2, y2, z2)]    

    if reverse:  
        start_color = dict_vertices[(y1, x1, z1)]
        end_color = dict_vertices[(y2, x2, z2)]

    delta_color = np.subtract(end_color, start_color)
    color_slope = np.divide(delta_color, (x2-x1))
    cur_color = np.add(start_color, np.multiply(delta_x, color_slope))
        
    if tot_points == 1:
        if reverse:
            points_to_fill.append((math.floor(cur_y + 0.5), cur_x, cur_z))
            dict_vertices[(math.floor(cur_y + 0.5), cur_x)] = tuple(cur_color)

        else:          
            points_to_fill.append((cur_x, math.floor(cur_y + 0.5), cur_z))
            dict_vertices[(cur_x, math.floor(cur_y + 0.5))] = tuple(cur_color)

        return points_to_fill, dict_vertices
        

    while (cur_x < x2):
        # if (cur_x <0 or cur_x > width or math.floor(cur_y + 0.5) <0 or math.floor(cur_y + 0.5) > height):
        #     break
        if reverse:
            points_to_fill.append((math.floor(cur_y + 0.5), cur_x, cur_z))
            dict_vertices[(math.floor(cur_y + 0.5), cur_x)] = tuple(cur_color)
 
        else:          
            points_to_fill.append((cur_x, math.floor(cur_y + 0.5), cur_z))
            dict_vertices[(cur_x, math.floor(cur_y + 0.5))] = tuple(cur_color)

        cur_color += np.multiply(delta_color, 1/(x2-x1))

        cur_x += 1
        cur_y += slope
        cur_z += slope_z
    return points_to_fill, dict_vertices


def create_line_color_step_y(point1, point2, dict_vertices, height, width):
    points_to_fill = []
    p1, p2 = np.array(point1, dtype=np.float64), np.array(point2, dtype=np.float64)

    # eight parts: xyzwrgba
    p1 = np.append(p1, list(dict_vertices[tuple(p1[:3])]))
    p2 = np.append(p2, list(dict_vertices[tuple(p2[:3])]))
    dp = p2 - p1
    d = abs(dp[1])

    # if the two points don't have the same y
    if d:
        delta = dp/d
        if p2[1]<p1[1]:
            delta = -delta
            p1, p2 = p2, p1
        s = (math.ceil(p1[1])-p1[1])/dp[1]
        q = p1+s*dp

        while q[1]<p2[1]: 

            points_to_fill.append(tuple(q.tolist())[:3])
            dict_vertices[tuple(q.tolist())[:3]] = tuple(q.tolist())[4:]
            q += delta

    return points_to_fill, dict_vertices


def check_counterclockwise(point1, point2, point3):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    x3, y3 = point3[0], point3[1]

    # res = x1*y2-y1*x2+y1*x3-x1*y3+x2*y3-x3*y2
    res = (y2-y1)*(x3-x2)-(x2-x1)*(y3-y2)
    # return res>0
    return res >0


# returns a list of colors
def set_color(input_of_rgb):
    res = []
    for c in input_of_rgb:
        if c <=0:
            res.append(0)
        elif c >= 1:
            res.append(255)
        else:
            # append int?
            res.append(c*255)
    return res


def quaternion_multiply(w1, x1, y1, z1, w2, x2, y2, z2):
    term1 = w1*w2 - x1*x2 - y1*y2 - z1*z2
    term2 = w1*x2 + x1*w2 + y1*z2 - z1*y2
    term3 = w1*y2 + y1*w2 + z1*x2 - x1*z2
    term4 = w1*z2 + z1*w2 + x1*y2 - y1*x2
    
    return np.array([term1, term2, term3, term4])


def calc_mvmatrix(origin, position, quaternion, scale):
    TO = np.matrix([[1,0,0,origin[0]], [0,1,0,origin[1]],[0,0,1,origin[2]], [0,0,0,1]])
    TP = np.matrix([[1,0,0,position[0]], [0,1,0,position[1]],[0,0,1,position[2]], [0,0,0,1]])
    
    if isinstance(quaternion[0], str):
        R = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        dict = {quaternion[0][0]: quaternion[1],quaternion[0][1]: quaternion[2],quaternion[0][2]: quaternion[3]}
        for orien in quaternion[0][::-1]:
            if orien == 'x':
                rx = float(dict[orien])*np.pi/180
                mat = np.matrix([[1,0,0,0],
                            [0,np.cos(rx),-np.sin(rx),0],
                            [0,np.sin(rx),np.cos(rx),0],
                            [0,0,0,1]])
                R = np.matmul(R, mat)

            elif orien == 'y':
                ry = float(dict[orien])*np.pi/180
                mat = np.matrix([[np.cos(ry),0,np.sin(ry),0],
                            [0,1,0,0],
                            [-np.sin(ry),0,np.cos(ry),0],
                            [0,0,0,1]])
                R = np.matmul(R, mat)

            elif orien == 'z':
                rz = float(dict[orien])*np.pi/180
                mat = np.matrix([ [np.cos(rz),-np.sin(rz),0,0],
                            [np.sin(rz),np.cos(rz),0,0],
                            [0,0,1,0],
                            [0,0,0,1]])
                R = np.matmul(R, mat)
    else:
        w, x, y, z = [val for val in quaternion]
        n = w**2 + x**2 + y**2 + z**2
        s = 0 if n == 0 else 2/n
        R = np.matrix([[1-s*(y**2+z**2),s*(x*y-z*w),s*(x*z+y*w),0], 
                        [s*(x*y+z*w),1-s*(x**2+z**2),s*(y*z-x*w),0],
                        [s*(x*z-y*w),s*(y*z+x*w),1-s*(x**2+y**2),0],
                        [0,0,0,1]])

    if len(scale) == 3:
        S = np.matrix([ [scale[0],0,0,0],
                    [0,scale[1],0,0],
                    [0,0,scale[2],0],
                    [0,0,0,1]])

    else:
        sx, sy, sz, w, x, y, z = scale
        q = np.array([w,x,y,z])
        q = q/np.linalg.norm(q)
        w, x, y, z = [val for val in q]
        n = w**2 + x**2 + y**2 + z**2
        s = 0 if n == 0 else 2/n
        R_S = np.matrix([[1-s*(y**2+z**2),s*(x*y-z*w),s*(x*z+y*w),0], 
                    [s*(x*y+z*w),1-s*(x**2+z**2),s*(y*z-x*w),0],
                    [s*(x*z-y*w),s*(y*z+x*w),1-s*(x**2+y**2),0],
                    [0,0,0,1]])

        S = np.matrix([[sx,0,0,0],
                    [0,sy,0,0],
                    [0,0,sz,0],
                    [0,0,0,1]])

        S = np.matmul(np.matmul(R_S,S),np.linalg.inv(R_S))

    mv_matrix = np.matmul(TO, np.matmul(TP, np.matmul(R, np.matmul(S, inv(TO)))))
    return mv_matrix
    
def ret_num_or_var(var_dict, var):
    try: a = float(var)
    except: a = var_dict[var]
    return a


def run(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines if line.rstrip() != '']
        lines = [line for line in lines if line.split()[0] in key_words]

    mega_info = lines[0].split()
    width = int(mega_info[1])
    height = int(mega_info[2])


    if mega_info[0] == 'pngs':

        prefix = mega_info[3]
        count = int(mega_info[4])
        l = count
        var = {'l': count}

        object_name_mapping = {}


        # i = 0
        for i in range(0, count):
            var['f'] = i
            image = Image.new("RGBA", (width, height), (0,0,0,0))

            do_all = True
            read = ''
            met_else = False
            has_camera = False

            for j in range(1, len(lines)):
                line = lines[j]
                info = line.split()
                op = info[0]

                if op == 'camera':
                    has_camera = True
                    cam_origin = np.array([0,0,0])
                    cam_scale = np.array([1,1,1])
                    cam_position = np.array([0,0,0])
                    cam_quaternion = np.array([1,0,0,0])

                    for line_inner in lines[j+1:]:
                        info_inner = line_inner.split()
                        op_inner = info_inner[0]
                        if op_inner == 'position':
                            if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                                cam_position = np.array([float(ret_num_or_var(var, info_inner[1])),float(ret_num_or_var(var, info_inner[2])), float(ret_num_or_var(var, info_inner[3]))])
             

                        elif op_inner == 'quaternion':
                            if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                                cam_quaternion = np.array([float(ret_num_or_var(var, info_inner[1])),float(ret_num_or_var(var, info_inner[2])), float(ret_num_or_var(var, info_inner[3])), float(ret_num_or_var(var, info_inner[4]))])
                    break

                elif op == 'add':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        a = ret_num_or_var(var, info[2])
                        b = ret_num_or_var(var, info[3])
                        var[str(info[1])] = a + b

                elif op == 'sub':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        a = ret_num_or_var(var, info[2])
                        b = ret_num_or_var(var, info[3])
                        var[str(info[1])] = a - b

                elif op == 'mul':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        a = ret_num_or_var(var, info[2])
                        b = ret_num_or_var(var, info[3])
                        var[str(info[1])] = a * b

                elif op == 'div':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        a = ret_num_or_var(var, info[2])
                        b = ret_num_or_var(var, info[3])
                        var[str(info[1])] = a / b

                elif op == 'pow':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        a = ret_num_or_var(var, info[2])
                        b = ret_num_or_var(var, info[3])
                        var[str(info[1])] = a ** b

                elif op == 'sin':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        a = ret_num_or_var(var, info[2])
                        var[str(info[1])] = np.sin(a*np.pi/180)

                elif op == 'cos':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        a = ret_num_or_var(var, info[2])
                        var[str(info[1])] = np.cos(a*np.pi/180)



            object_name = ""
            parent = ""

            origin = np.array([0,0,0])
            scale = np.array([1,1,1])
            position = np.array([0,0,0])
            quaternion = np.array([1,0,0,0])
            var = {'l': count}

            list_vertices = []
            dict_vertices = {}
            color_cur = (1,1,1)
            #key: (x,y), value:z
            depth_buffer = {}
            for x in range (0, width):
                for y in range(0, height):
                    depth_buffer[(x,y)] = 1
            # mv_matrix = np.identity(4)
            mv_matrix = np.matrix([[1,0,0,0], [0,1,0,0],[0,0,1,0], [0,0,0,1]])
            proj_matrix = np.matrix([[1,0,0,0], [0,1,0,0],[0,0,1,0], [0,0,0,1]])

            var['f'] = i

            for line in lines[1:]:
                filename = str(prefix) + str(str(i).zfill(3)) + '.png'
                info = line.split()
                op = info[0]
                if op == 'xyz':
                    #x, y, z, w, r, g, b
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        list_vertices.append((float(ret_num_or_var(var, info[1])),float(ret_num_or_var(var, info[2])), float(ret_num_or_var(var, info[3])) ,1.0, *color_cur))
                        dict_vertices[(float(ret_num_or_var(var, info[1])),float(ret_num_or_var(var, info[2])), float(ret_num_or_var(var, info[3])), 1.0)] = color_cur 
                
                elif op == 'color':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        color_cur = np.array([float(ret_num_or_var(var, info[1])),float(ret_num_or_var(var, info[2])), float(ret_num_or_var(var, info[3]))])

                elif op == 'loadp':
                    all_num = [float(a) for a in info[1:]]
                    proj_matrix = np.matrix([all_num[0:4], all_num[4:8], all_num[8:12], all_num[12:]])

                elif op == 'object':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        list_vertices = []
                        dict_vertices = {}
                        object_name, parent = str(info[1]), str(info[2])
                        origin = np.array([0,0,0])
                        scale = np.array([1,1,1])
                        position = np.array([0,0,0])
                        quaternion = np.array([1,0,0,0])
                        first_trif = True

                elif op == 'position':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        a = ret_num_or_var(var, info[2])
                        position = np.array([float(ret_num_or_var(var, info[1])),float(ret_num_or_var(var, info[2])), float(ret_num_or_var(var, info[3]))])

                elif op == 'quaternion':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        quaternion = np.array([float(ret_num_or_var(var, info[1])),float(ret_num_or_var(var, info[2])), float(ret_num_or_var(var, info[3])), float(ret_num_or_var(var, info[4]))])

                elif op == 'origin':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        origin = np.array([float(info[1]),float(info[2]),float(info[3])])

                elif op == 'scale':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        scale = np.array([float(ret_num_or_var(var, info[1])),float(ret_num_or_var(var, info[2])), float(ret_num_or_var(var, info[3]))])
                    # print('scale:',scale)

                elif op == 'anyscale':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        scale = np.array([float(ret_num_or_var(var, info[1])),float(ret_num_or_var(var, info[2])), float(ret_num_or_var(var, info[3])), float(ret_num_or_var(var, info[4])), float(ret_num_or_var(var, info[5])), float(ret_num_or_var(var, info[6])), float(ret_num_or_var(var, info[7]))])

                elif op == 'euler':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        quaternion = np.array([str(info[1]),float(ret_num_or_var(var, info[2])), float(ret_num_or_var(var, info[3])), float(ret_num_or_var(var, info[4]))])

                elif op == 'iflt':
                    x = float(ret_num_or_var(var, info[1]))
                    y = float(ret_num_or_var(var, info[2]))
                    if x < y:
                        read = 'front'
                        met_else = False
                        do_all = False
                    else:
                        read = 'end'
                        met_else = False
                        do_all = False

                elif op == 'else':
                    met_else = True

                elif op == 'fi':
                    do_all = True

                elif op == 'add':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        a = ret_num_or_var(var, info[2])
                        b = ret_num_or_var(var, info[3])
                        var[str(info[1])] = a + b

                elif op == 'sub':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        a = ret_num_or_var(var, info[2])
                        b = ret_num_or_var(var, info[3])
                        var[str(info[1])] = a - b

                elif op == 'mul':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        a = ret_num_or_var(var, info[2])
                        b = ret_num_or_var(var, info[3])
                        var[str(info[1])] = a * b

                elif op == 'div':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        a = ret_num_or_var(var, info[2])
                        b = ret_num_or_var(var, info[3])
                        var[str(info[1])] = a / b

                elif op == 'pow':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        a = ret_num_or_var(var, info[2])
                        b = ret_num_or_var(var, info[3])
                        var[str(info[1])] = a ** b

                elif op == 'sin':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        a = ret_num_or_var(var, info[2])
                        var[str(info[1])] = np.sin(a*np.pi/180)

                elif op == 'cos':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        a = ret_num_or_var(var, info[2])
                        var[str(info[1])] = np.cos(a*np.pi/180)


                elif op == 'trif':
                    if (read == 'front' and not met_else) or (read == 'end' and met_else) or do_all:
                        if has_camera:
                            camera_matrix = calc_mvmatrix(cam_origin, cam_position, cam_quaternion, cam_scale)
                            camera_matrix = inv(camera_matrix)

                        if first_trif:
                            mv_matrix = calc_mvmatrix(origin, position, quaternion, scale)
                            
                            if parent != 'world':
                                parent_matrix = object_name_mapping[parent]
                                mv_matrix = np.matmul(parent_matrix, mv_matrix)
                        
                            object_name_mapping[object_name] = mv_matrix
        

                        first_trif = False

                        point1 = np.array(list_vertices[int(info[1])][0:4] if int(info[1]) < 0 else list_vertices[int(info[1])-1][0:4])
                        point2 = np.array(list_vertices[int(info[2])][0:4] if int(info[2]) < 0 else list_vertices[int(info[2])-1][0:4])
                        point3 = np.array(list_vertices[int(info[3])][0:4] if int(info[3]) < 0 else list_vertices[int(info[3])-1][0:4])

                        points = [point1, point2, point3]


                        points_res = []
                        for point in points:
                            # model view transformation
                            point1 = np.matmul(mv_matrix, point).tolist()
                            point1 = [i for w in point1 for i in w]

                            if has_camera:
                                point1 = np.matmul(camera_matrix, point1).tolist()
                                point1 = [i for w in point1 for i in w]

                            # projection matrix
                            point1 = np.matmul(proj_matrix, point1).tolist()
                            point1 = [i for w in point1 for i in w]
                            # divide x, y, and z by w
                            point1[0] /= point1[3]
                            point1[1] /= point1[3]
                            point1[2] /= point1[3]

                            #viewport transformation
                            point1[0] = (point1[0]+ 1) * width /2
                            point1[1] = (point1[1]+ 1) * height /2

                            points_res.append(point1)


                        point1 = tuple(points_res[0])
                        point2 = tuple(points_res[1])
                        point3 = tuple(points_res[2])
                        

                        r, g, b = set_color(list(color_cur))

                        # use x,y,z,w as key
                        dict_vertices[point1[0:3]] = (r, g, b, 255)
                        dict_vertices[point2[0:3]] = (r, g, b, 255)
                        dict_vertices[point3[0:3]] = (r, g, b, 255)

                        points_to_fill_12, dict_vertices_temp = create_line_color_step_y(point1, point2, dict_vertices, height, width)
                        dict_vertices.update(dict_vertices_temp)
                        points_to_fill_13, dict_vertices_temp = create_line_color_step_y(point1, point3, dict_vertices, height, width)
                        dict_vertices.update(dict_vertices_temp)
                        points_to_fill_23, dict_vertices_temp = create_line_color_step_y(point2, point3, dict_vertices, height, width)
                        dict_vertices.update(dict_vertices_temp)

                        all_y = list([a[1] for a in points_to_fill_12] +[a[1] for a in points_to_fill_13] + [a[1] for a in points_to_fill_23])
                        all_y = list(set(all_y))
                        all_y.sort()

                        # print(len(all_y))
                        for y in all_y:
                        
                            all_x = list([a[0] for a in points_to_fill_12 if a[1] == y] +[a[0] for a in points_to_fill_13 if a[1] == y] + [a[0] for a in points_to_fill_23 if a[1] == y])
                            all_x = [x for x in all_x if x is not None]

                            start_x = min(all_x)
                            end_x = max(all_x)
                            if math.ceil(start_x) < end_x:
                                
                                z_start = list([a[2] for a in points_to_fill_12 if a[1] == y and a[0]==start_x] +[a[2] for a in points_to_fill_13 if a[1] == y and a[0]==start_x] + [a[2] for a in points_to_fill_23 if a[1] == y and a[0]==start_x])[0]
                                z_end = list([a[2] for a in points_to_fill_12 if a[1] == y and a[0]==end_x] +[a[2] for a in points_to_fill_13 if a[1] == y and a[0]==end_x] + [a[2] for a in points_to_fill_23 if a[1] == y and a[0]==end_x])[0]
                        
                                points_to_fill, dict_vertices_temp = create_line_color((start_x, y, z_start), (end_x, y, z_end), dict_vertices)
                                dict_vertices.update(dict_vertices_temp)

                                for point in points_to_fill:
                                    if 0 <= int(point[0]) and int(point[0]) < width and 0 <= int(point[1]) and int(point[1]) < height:

                                        z = point[2]
                                        if z <=1 and z>=0 and z <= depth_buffer[(int(point[0]),int(point[1]))]:
                                            r, g, b, alpha = dict_vertices[(int(point[0]),int(point[1]))]
                                            image.im.putpixel((int(point[0]),int(point[1])), (int(r), int(g), int(b), int(alpha)))
                                            depth_buffer[(int(point[0]),int(point[1]))] = z
                            
            image.save(filename)


if __name__ == "__main__":
    run(sys.argv[1])
