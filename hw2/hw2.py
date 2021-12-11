from PIL import Image
import os
import math
import numpy as np
import sys

# key_words = ['xyrgb', 'xyc', 'png', 'pngs', 'trig', 'linec', 'lineg', 'tric', 'xyrgba', 'trica', 'polyec', 'polynz', 'fann', 'stripn', 'xyz', 'color', 'trif', 'loadmv', 'loadp', 'xyzw', 'frustum']
key_words = ['png', 'xyz', 'color', 'trif', 'loadmv', 'loadp', 'xyzw', 'frustum', 'ortho', 'translate', 'rotatex', 'rotatey', 'rotatez', 'scale', 'multmv', 'trig', 'cull', 'lookat']
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
        x_small_to_large = p2[0] - p1[0]
        while q[1]<p2[1]: 
            #  dir: \
            if x_small_to_large >=0:
                if (q[1]>height):
                    break
                elif (q[0]>width):
                    q[0]=width


                elif (q[1] < 0):
                    s = (0-q[1])/dp[1]
                    q += s*dp
                else:
                    points_to_fill.append(tuple(q.tolist())[:3])
                    dict_vertices[tuple(q.tolist())[:3]] = tuple(q.tolist())[4:]
                    q += delta
            #dir: /
            else:
                if (q[1]>height):
                    break
                elif q[0] <0:
                    q[0] = 0

                elif (q[1] < 0):
                    s = (0-q[1])/dp[1]
                    q += s*dp
                else:
                    points_to_fill.append(tuple(q.tolist())[:3])
                    dict_vertices[tuple(q.tolist())[:3]] = tuple(q.tolist())[4:]
                    q += delta

    # returns a tuple of four elements: x, y, z

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


def run(filename):
    potential_overlap = {}

    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines if line.rstrip() != '']
        lines = [line for line in lines if line.split()[0] in key_words]

    mega_info = lines[0].split()
    width = int(mega_info[1])
    height = int(mega_info[2])


    if mega_info[0] == 'png':
        list_vertices = []
        # contains two forms of data: 1. key: (x,y,z,w); 2. key: (x,y). both have (r,g,b,alpha) as value
        dict_vertices = {}

        filename = mega_info[3]
        color_cur = (1,1,1)
        #key: (x,y), value:z
        depth_buffer = {}
        for x in range (0, width):
            for y in range(0, height):
                depth_buffer[(x,y)] = 1
        # mv_matrix = np.identity(4)
        mv_matrix = np.matrix([[1,0,0,0], [0,1,0,0],[0,0,1,0], [0,0,0,1]])
        proj_matrix = np.matrix([[1,0,0,0], [0,1,0,0],[0,0,1,0], [0,0,0,1]])
        cull = False


        image = Image.new("RGBA", (width, height), (0,0,0,0))
        for line in lines[1:]:

            info = line.split()
            op = info[0]

            #starting hw2
            # colors specified in range (0,1)
            if op == 'xyz':
                #x, y, z, w, r, g, b
                list_vertices.append((float(info[1]),float(info[2]), float(info[3]) ,1.0, *color_cur))
                dict_vertices[(float(info[1]),float(info[2]), float(info[3]), 1.0)] = color_cur 

            elif op == 'xyzw':
                list_vertices.append((float(info[1]),float(info[2]), float(info[3]) , float(info[4]), 1, 1, 1))
                dict_vertices[(float(info[1]),float(info[2]), float(info[3]), float(info[4]))] = (1,1,1)   


            elif op == 'color':
                color_cur = (float(info[1]),float(info[2]),float(info[3]))

            elif op == 'cull':
                cull = True


            elif op == 'loadmv':
                all_num = [float(a) for a in info[1:]]
                mv_matrix = np.matrix([all_num[0:4], all_num[4:8], all_num[8:12], all_num[12:]])


            elif op == 'loadp':
                all_num = [float(a) for a in info[1:]]
                proj_matrix = np.matrix([all_num[0:4], all_num[4:8], all_num[8:12], all_num[12:]])


            elif op == 'translate':
                delta_x, delta_y, delta_z = float(info[1]),float(info[2]), float(info[3])
                mv_matrix = np.matmul(mv_matrix, np.matrix([[1,0,0,delta_x], [0,1,0,delta_y],[0,0,1,delta_z], [0,0,0,1]]))


            elif op == 'rotatex':
                degree = math.radians(float(info[1]))
                cos = math.cos(degree)
                sin = math.sin(degree)
                mv_matrix = np.matmul(mv_matrix, np.matrix([[1,0,0,0], [0,cos,-1*sin,0],[0,sin,cos,0], [0,0,0,1]]))


            elif op == 'rotatey':
                degree = math.radians(float(info[1]))
                cos = math.cos(degree)
                sin = math.sin(degree)
                mv_matrix = np.matmul(mv_matrix, np.matrix([[cos,0,sin,0], [0,1,0,0],[-1*sin,0,cos,0], [0,0,0,1]]))
        

            elif op == 'rotatez':
                degree = math.radians(float(info[1]))
                cos = math.cos(degree)
                sin = math.sin(degree)
                mv_matrix = np.matmul(mv_matrix, np.matrix([[cos,-1*sin,0,0], [sin,cos,0,0],[0,0,1,0], [0,0,0,1]]))
            
            elif op =='scale':
                sx, sy, sz = float(info[1]),float(info[2]), float(info[3]) 
                mv_matrix = np.matmul(mv_matrix, np.matrix([[sx,0,0,0], [0,sy,0,0],[0,0,sz,0], [0,0,0,1]]))


            elif op =='multmv':
                all_num = [float(a) for a in info[1:]]
                new_mv_matrix = np.matrix([all_num[0:4], all_num[4:8], all_num[8:12], all_num[12:]])
                mv_matrix = np.matmul(mv_matrix, new_mv_matrix)


            elif op == 'ortho':
                l, r, b, t, n, f = float(info[1]),float(info[2]), float(info[3]) , float(info[4]), float(info[5]), float(info[6])
                n = 2*n-f
                term_a = (r+l)/(r-l)
                term_b = (t+b)/(t-b)
                term_c = -(f+n)/(f-n)
                term_d = -2/(f-n)
                proj_matrix = np.matrix([[2/(r-l),0,0, -1*term_a], [0,2/(t-b),0, -1*term_b],[0,0,term_d,term_c], [0,0,0,1]])

                # proj_matrix = np.matrix([[2*n/(r-l),0,term_a,0], [0,2*n/(t-b),term_b,0],[0,0,term_c,term_d], [0,0,-1,0]])


            elif op == 'frustum':
                l, r, b, t, n, f = float(info[1]),float(info[2]), float(info[3]) , float(info[4]), float(info[5]), float(info[6])
                term_a = (r+l)/(r-l)
                term_b = (t+b)/(t-b)
                term_c = -(f+n)/(f-n)
                term_d = -2*(f*n)/(f-n)
                proj_matrix = np.matrix([[2*n/(r-l),0,term_a,0], [0,2*n/(t-b),term_b,0],[0,0,term_c,term_d], [0,0,-1,0]])

            # concept reference: https://www2.cs.sfu.ca/~haoz/teaching/htmlman/lookat.html
            elif op =='lookat':
                eye, center, up = np.array(list_vertices[int(info[1])-1][0:3]), np.array(list_vertices[int(info[2])-1][0:3]), (float(info[3]) , float(info[4]), float(info[5]))
            
                forward = center - eye
                forward = forward/np.linalg.norm(forward)
                up = up/np.linalg.norm(up)
                left = np.cross(forward, up)
                left = left/np.linalg.norm(left)
                up = np.cross(left, forward)
                
                m1 = np.matrix([[*left,0],
                                [*up,0],
                                [-1*forward[0], -1*forward[1], -1*forward[2], 0],
                                [0,0,0,1]])

                m2 = np.matrix([[1,0,0,-1*eye[0]], 
                                [0,1,0,-1*eye[1]],
                                [0,0,1,-1*eye[2]], 
                                [0,0,0,1]])
                mv_matrix = np.matmul(m1, m2)



            elif op == 'trif':
                # four components: x,y,z,w
                point1 = np.array(list_vertices[int(info[1])][0:4] if int(info[1]) < 0 else list_vertices[int(info[1])-1][0:4])
                point2 = np.array(list_vertices[int(info[2])][0:4] if int(info[2]) < 0 else list_vertices[int(info[2])-1][0:4])
                point3 = np.array(list_vertices[int(info[3])][0:4] if int(info[3]) < 0 else list_vertices[int(info[3])-1][0:4])

                points = [point1, point2, point3]
                points_res = []
                for point in points:
                    # model view transformation
                    point1 = np.matmul(mv_matrix, point).tolist()
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
                
                if cull:
                    if check_counterclockwise(point1, point2, point3):

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
                                        r, g, b, alpha = dict_vertices[(int(point[0]),int(point[1]))]
                                        image.im.putpixel((int(point[0]),int(point[1])), (int(r), int(g), int(b), int(alpha)))



                else:
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


                        




            elif op == 'trig':
                point1 = np.array(list_vertices[int(info[1])][0:4] if int(info[1]) < 0 else list_vertices[int(info[1])-1][0:4])
                point2 = np.array(list_vertices[int(info[2])][0:4] if int(info[2]) < 0 else list_vertices[int(info[2])-1][0:4])
                point3 = np.array(list_vertices[int(info[3])][0:4] if int(info[3]) < 0 else list_vertices[int(info[3])-1][0:4])
                points = [point1, point2, point3]
                points_res = []

                color_res = [set_color(list(dict_vertices[tuple(point1)])), set_color(list(dict_vertices[tuple(point2)])),set_color(list(dict_vertices[tuple(point3)]))]
                for point in points:
                    # model view transformation
                    point1 = np.matmul(mv_matrix, point).tolist()
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

                # use x,y,z,w as key
                dict_vertices[point1[0:3]] = (*color_res[0], 255)
                dict_vertices[point2[0:3]] = (*color_res[1], 255)
                dict_vertices[point3[0:3]] = (*color_res[2], 255)

                points_to_fill_12, dict_vertices_temp = create_line_color_step_y(point1, point2, dict_vertices, height, width)
                dict_vertices.update(dict_vertices_temp)
                points_to_fill_13, dict_vertices_temp = create_line_color_step_y(point1, point3, dict_vertices, height, width)
                dict_vertices.update(dict_vertices_temp)
                points_to_fill_23, dict_vertices_temp = create_line_color_step_y(point2, point3, dict_vertices, height, width)
                dict_vertices.update(dict_vertices_temp)

                all_y = list([a[1] for a in points_to_fill_12] +[a[1] for a in points_to_fill_13] + [a[1] for a in points_to_fill_23])
                all_y = list(set(all_y))
                all_y.sort()

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
