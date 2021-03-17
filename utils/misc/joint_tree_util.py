
def makedepth(child):
    def makedepth_rec(child, depth, node):
        for x in child[node]:
            depth[x] = depth[node] + 1
        for x in child[node]:
            makedepth_rec(child, depth, x)
        return 
    depth = [0 for x in range(len(child))]
    root = 0
    makedepth_rec(child, depth, root)
    return depth
