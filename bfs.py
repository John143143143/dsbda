g = {
    0 : [1,2],
    1 : [0,3,4],
    2 : [3,0],
    3 : [2,1,4],
    4 : [1,3]
}

def dfs(g,s):
    vis[s] = 1
    print(s)
    for c in g[s]:
        if not vis[c]:
            dfs(g,c)

vis = [0] * 5

dfs(g,0)