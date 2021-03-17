class ObjLoader(object):
    def __init__(self, fileName, skip_face=False, process_face=True):
        self.vertices = []
        self.faces = []
        ##
        vertex_processing=False
        try:
            f = open(fileName)
            for line in f:
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                    vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                    self.vertices.append(vertex)
                    vertex_processing = True

                elif line[0] == "f":
                    string = line.replace("//", "/")
                    ##
                    i = string.find(" ") + 1
                    face = []
                    for item in range(string.count(" ")):
                        if string.find(" ", i) == -1:
                            fragment = string[i:-1]
                            if process_face:
                                fragment = int(fragment.split("/")[0]) - 1
                            face.append(fragment)
                            break
                        fragment = string[i:string.find(" ", i)]
                        if process_face:
                            fragment = int(fragment.split("/")[0]) - 1
                        face.append(fragment)
                        i = string.find(" ", i) + 1
                    ##
                    self.faces.append(tuple(face))
                else:
                    # single mesh assumption
                    if skip_face and vertex_processing:
                        break


            f.close()
        except IOError:
            print(".obj file not found.")


def face2edge(faces:list):
    # faces = [(1,2,3), (3,4,5), (6,7,8)]
    e = []
    for f in faces:
        e += [[f[0], f[1]], [f[1],f[2]], [f[2],f[0]], [f[1], f[0]], [f[0], f[2]], [f[2], f[1]]]
    return e