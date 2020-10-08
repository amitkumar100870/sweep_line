import cv2
import sys
import json
from math import isinf
import matplotlib.pyplot as plt

# Debug function for lists
def show(L: list):
    n = len(L)
    for i in range(n):
        print(L[i])
    print()

# Plotting points and line
def plot(lines , points=None):
    for i in lines:
        lx = [i.l.x]
        ly = [i.l.y]
        plt.scatter(lx,ly, c='b', label="Left Points")
        rx = [i.r.x]
        ry = [i.r.y]
        plt.scatter(rx,ry, c='g', label="Right Points")
        plt.plot([i.l.x , i.r.x] , [i.l.y , i.r.y] , 'k')
    if points is not None:
        for i in points:
            x = [i.x]
            y = [i.y]
            plt.scatter(x,y, c='r', label="Left Points")

    # plt.legend(loc='upper left')
    plt.show()

# Get Input from Input File
def getInput():
    f = open('Input' , 'r')
    T = json.load(f)
    return T

# Point -> (x,y)
class Point:
    def __init__(self, P):
        self.x = P[0]
        self.y = P[1]

    def __repr__(self):
        return "({},{})".format(self.x , self.y)

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        else:
            return False

# Line ->  P(x1,y1) ----------------------- Q(x2,y2)
class Line:
    def __init__(self , l , r):
        self.l = l
        self.r = r
        if self.l.x - self.r.x == 0:
            self.m = float('inf')
        else: 
            self.m = (self.l.y - self.r.y)/(self.l.x - self.r.x)
        if isinf(self.m):
            self.c = self.l.x
        else:
            self.c = self.l.y - (self.m*self.l.x)
    
    def getm(self):
        return self.m
    
    def getc(self):
        return self.c

    def __repr__(self):
        return "{} <-> {}".format(self.l , self.r)

# Checks collinearity
def on(a,b,c):
    if b.x <= max(a.x , c.x) and b.x >= min(a.x , c.x) and b.y <= max(a.y , c.y) and b.y >= min(a.y , c.y):
        return True
    return False

# -ve is ccw, +ve is cw, 0 is collinear
def ccw(a,b,c):
    # Change to float if error
    d = (b.y - a.y)*(c.x - b.x) - (c.y - b.y)*(b.x - a.x)
    if d == 0:
        return 0
    elif d>0:
        return 1
    elif d<0:
        return -1

# Checks line intersect
def intersect(p1,q1,p2,q2): 
      
    # Find the 4 orientations required for  
    # the general and special cases 
    o1 = ccw(p1, q1, p2) 
    o2 = ccw(p1, q1, q2) 
    o3 = ccw(p2, q2, p1) 
    o4 = ccw(p2, q2, q1) 
  
    # General case 
    if ((o1 != o2) and (o3 != o4)): 
        return True
  
    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1 
    if ((o1 == 0) and on(p1, p2, q1)): 
        return True
  
    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
    if ((o2 == 0) and on(p1, q2, q1)): 
        return True
  
    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
    if ((o3 == 0) and on(p2, p1, q2)): 
        return True
  
    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
    if ((o4 == 0) and on(p2, q1, q2)): 
        return True
    return False

def createLines(Ti):
    L = []
    for i in Ti:
        a = [Point(i[0]) , Point(i[1])]
        a = sorted(sorted(a , key = lambda x: x.y) , key=lambda x: x.x)
        L.append(Line(a[0] , a[1]))
    return L

# Node for Tree
class Node():
    def __init__(self, data):
        self.data = data
        self.parent = None
        self.left = None
        self.right = None
        self.color = 1
    
    def __repr__(self):
        return "({})".format(self.data)

class RedBlackTree():
    def __init__(self):
        self.TNULL = Node(0)
        self.TNULL.color = 0
        self.TNULL.left = None
        self.TNULL.right = None
        self.root = self.TNULL

    # Preorder
    def pre_order_helper(self, node):
        if node != TNULL:
            sys.stdout.write(node.item + " ")
            self.pre_order_helper(node.left)
            self.pre_order_helper(node.right)

    # Inorder
    def in_order_helper(self, node):
        if node != TNULL:
            self.in_order_helper(node.left)
            sys.stdout.write(node.item + " ")
            self.in_order_helper(node.right)

    # Postorder
    def post_order_helper(self, node):
        if node != TNULL:
            self.post_order_helper(node.left)
            self.post_order_helper(node.right)
            sys.stdout.write(node.item + " ")

    # Search the tree
    def search_tree_helper(self, node, key):
        # print("Key: {}".format(key))
        if node == self.TNULL or key[0] == node.data[0]:
            return node

        if key[0].y < node.item[0].y:
            return self.search_tree_helper(node.left, key)
        return self.search_tree_helper(node.right, key)

    # Balancing the tree after deletion
    def delete_fix(self, x):
        while x != self.root and x.color == 0:
            if x == x.parent.left:
                s = x.parent.right
                if s.color == 1:
                    s.color = 0
                    x.parent.color = 1
                    self.left_rotate(x.parent)
                    s = x.parent.right

                if s.left.color == 0 and s.right.color == 0:
                    s.color = 1
                    x = x.parent
                else:
                    if s.right.color == 0:
                        s.left.color = 0
                        s.color = 1
                        self.right_rotate(s)
                        s = x.parent.right

                    s.color = x.parent.color
                    x.parent.color = 0
                    s.right.color = 0
                    self.left_rotate(x.parent)
                    x = self.root
            else:
                s = x.parent.left
                if s.color == 1:
                    s.color = 0
                    x.parent.color = 1
                    self.right_rotate(x.parent)
                    s = x.parent.left

                if s.right.color == 0 and s.right.color == 0:
                    s.color = 1
                    x = x.parent
                else:
                    if s.left.color == 0:
                        s.right.color = 0
                        s.color = 1
                        self.left_rotate(s)
                        s = x.parent.left

                    s.color = x.parent.color
                    x.parent.color = 0
                    s.left.color = 0
                    self.right_rotate(x.parent)
                    x = self.root
        x.color = 0

    def __rb_transplant(self, u, v):
        if u.parent == None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    # Node deletion
    def delete_node_helper(self, node, key):
        z = self.TNULL
        while node != self.TNULL:
            if node.item == key:
                z = node

            if node.item[0].y <= key[0].y:
                node = node.right
            else:
                node = node.left

        if z == self.TNULL:
            print("Cannot find key in the tree")
            return

        y = z
        y_original_color = y.color
        if z.left == self.TNULL:
            x = z.right
            self.__rb_transplant(z, z.right)
        elif (z.right == self.TNULL):
            x = z.left
            self.__rb_transplant(z, z.left)
        else:
            y = self.minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent == z:
                x.parent = y
            else:
                self.__rb_transplant(y, y.right)
                y.right = z.right
                y.right.parent = y

            self.__rb_transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
        if y_original_color == 0:
            self.delete_fix(x)

    # Balance the tree after insertion
    def fix_insert(self, k):
        while k.parent.color == 1:
            if k.parent == k.parent.parent.right:
                u = k.parent.parent.left
                if u.color == 1:
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        k = k.parent
                        self.right_rotate(k)
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self.left_rotate(k.parent.parent)
            else:
                u = k.parent.parent.right

                if u.color == 1:
                    u.color = 0
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        k = k.parent
                        self.left_rotate(k)
                    k.parent.color = 0
                    k.parent.parent.color = 1
                    self.right_rotate(k.parent.parent)
            if k == self.root:
                break
        self.root.color = 0

    # Printing the tree
    def __print_helper(self, node, indent, last):
        if node != self.TNULL:
            sys.stdout.write(indent)
            if last:
                sys.stdout.write("R----")
                indent += "     "
            else:
                sys.stdout.write("L----")
                indent += "|    "

            s_color = "RED" if node.color == 1 else "BLACK"
            print(str(node.item) + "(" + s_color + ")")
            self.__print_helper(node.left, indent, False)
            self.__print_helper(node.right, indent, True)

    def preorder(self):
        self.pre_order_helper(self.root)

    def inorder(self):
        self.in_order_helper(self.root)

    def postorder(self):
        self.post_order_helper(self.root)

    def searchTree(self, k):
        return self.search_tree_helper(self.root, k)

    def minimum(self, node):
        while node.left != self.TNULL:
            node = node.left
        return node

    def maximum(self, node):
        while node.right != self.TNULL:
            node = node.right
        return node

    def successor(self, x):
        if (x.right != self.TNULL):
            return self.minimum(x.right)

        y = x.parent

        if y is None:
            return None

        while y is not None and y != self.TNULL and x == y.right:
            x = y
            y = y.parent
        return y

    def predecessor(self,  x):
        if (x.left != self.TNULL):
            return self.maximum(x.left)

        y = x.parent

        if y is None:
            return None

        # print(y , type(y))
        while y is not None and y != self.TNULL and x == y.left:
            x = y
            y = y.parent
        return y

    def left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.TNULL:
            y.left.parent = x

        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.TNULL:
            y.right.parent = x

        y.parent = x.parent
        if x.parent == None:
            self.root = y
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def insert(self, key):
        node = Node(key)
        node.parent = None
        node.item = key
        node.left = self.TNULL
        node.right = self.TNULL
        node.color = 1

        y = None
        x = self.root

        while x != self.TNULL:
            y = x
            if node.item[0].y < x.item[0].y:
                x = x.left
            else:
                x = x.right

        node.parent = y
        if y == None:
            self.root = node
        elif node.item[0].y < y.item[0].y:
            y.left = node
        else:
            y.right = node

        if node.parent == None:
            node.color = 0
            return

        if node.parent.parent == None:
            return

        self.fix_insert(node)

    def get_root(self):
        return self.root

    def delete_node(self, item):
        self.delete_node_helper(self.root, item)

    def print_tree(self):
        self.__print_helper(self.root, "", True)

def sweep(L):
    # All points, left, right
    P = []
    l = []
    r = []
    count = 0

    # Output Array
    Lf = []

    # Sorting Line segments by X coordinate
    L = sorted(sorted(L , key=lambda p: p.l.y) , key = lambda p: p.l.x)
    # print(L)

    # Creating Event Points
    for i in L:
        P.append([i.l , count , 0])
        P.append([i.r , count , 1])
        count+=1

    # Sorting Event points by X-axis
    P = sorted(sorted(P , key=lambda p: p[0].y) , key = lambda p: p[0].x)
    show(P)
    # Initialising tree
    tree = RedBlackTree()

    # If error check here -> Line 43
    # i -> [(x,y) , N_th line segment , left(0) / right(1)]
    for i in range(0 , len(P)):
        # tree.print_tree()
        # If left point
        if P[i][2] == 0:
            tree.insert(P[i])
            current = tree.searchTree(P[i])
            pred = tree.predecessor(current)
            succ = tree.successor(current)

            # Find predecessor and check if they intersect
            if pred is not None:
                # print(L[pred.data[1]].l , L[pred.data[1]].r , L[current.data[1]].l , L[current.data[1]].r)
                if intersect(L[pred.data[1]].l , L[pred.data[1]].r , L[current.data[1]].l , L[current.data[1]].r):
                    Lf.append([L[pred.data[1]] , L[current.data[1]]])

            #Find successor and check if they intersect
            if succ is not None:
                # print((L[current.data[1]].l , L[current.data[1]].r , L[succ.data[1]].l , L[succ.data[1]].r))
                if intersect(L[current.data[1]].l , L[current.data[1]].r , L[succ.data[1]].l , L[succ.data[1]].r):
                    Lf.append([L[current.data[1]] , L[succ.data[1]]])
        
        elif P[i][2] == 1:
            print(L[P[i][1]].l)
            current = tree.searchTree([L[P[i][1]].l , i , 0])
            pred = tree.predecessor(current)
            succ = tree.successor(current)
            if pred is not None and succ is not None:
                print((L[pred.data[1]].l , L[pred.data[1]].r , L[succ.data[1]].l , L[succ.data[1]].r))
                if intersect(L[pred.data[1]].l , L[pred.data[1]].r , L[succ.data[1]].l , L[succ.data[1]].r):
                    Lf.append([L[pred.data[1]] , L[succ.data[1]]])
            
            tree.delete_node([L[P[i][1]].l , P[i][1] , 0])
    
    return Lf

# def interP(L1, L2):
#     X = (L1.l.x - L1.r.x , L2.l.x - L2.r.x)
#     Y = (L1.l.y - L1.r.y , L2.l.y - L2.r.y)
#     # xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
#     # ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

#     def det(a, b):
#         return a[0] * b[1] - a[1] * b[0]

#     div = det(X,Y)
#     if div == 0:
#        raise Exception('lines do not intersect')

#     l1 = [[L1.l.x , L1.l.y] , [L1.r.x , L1.r.y]]
#     l2 = [[L2.l.x , L2.l.y] , [L2.r.x , L2.r.y]]

#     d = (det(l1[0] , l1[1]), det(l2[0] , l1[1]))
#     x = det(d, X) / div
#     y = det(d, Y) / div
#     return Point([x, y])

# print line_intersection((A, B), (C, D))

def coff(L1):
    p1 = [L1.l.x , L1.l.y]
    p2 = [L1.r.x , L1.r.y]
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return [A, B, -C]

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return Point([x,y])
    else:
        return False

if __name__ == "__main__":
    
    # print(intersect(Point([1,2]) , Point([2,3]) , Point([1,3]) , Point([2,2])))
    print("\n[INFO] --- Getting Input\n")
    T = getInput()

    # ith Test case
    for i in range(len(T)):
        # print(T[i])
        
        print("\n[INFO] --- Creating Lines\n")
        L = createLines(T[i])
        show(L)
        
        # print("\n[INFO] --- PLotting Lines\n")
        # plot(L)

        print("\n[INFO] --- Starting Sweep\n")
        I = sweep(L)
        for i in I:
            print(i)

        P = []
        for i in I:
            P.append(intersection(coff(i[0]) , coff(i[1])))
        plot(L , P)

# CPU times. -> 100x