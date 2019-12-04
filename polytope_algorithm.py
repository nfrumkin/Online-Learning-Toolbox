import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space, qr
import itertools
import copy
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import imageio

# hyperparams
min_val = -1
max_val = 1
num_pts = 500
d = 1 # dimension is in terms of x's --> x,y is 1d
epsilon = 1e-9
debug = False

# generate data
np.random.seed(1)
X = np.array([np.random.uniform(min_val, max_val, d) for i in range(0,num_pts)])
X[0] = min_val
X[1] = max_val
# specify covariance matrix
# cov = np.eye(d,d)
# y = np.diag(0.5*np.multiply(np.multiply(X.T,cov),X))
y = X**2

def get_point(n):
    return np.hstack([y[n], X[n]])

def is_degenerate(init_points, new_point):
    diff_mat = np.subtract(init_points, new_point)
    rank = np.linalg.matrix_rank(diff_mat)
    return rank <= d

def check_endpoint(x, current_dom):
    leftmost = False
    rightmost = False
    dims = []
    for i in range(0,d):
        x_i = x[i]
        if x_i <= current_dom[i,0]:
            current_dom[i,0] = x_i
            leftmost = True
            dims.append(i)
        elif x_i >= current_dom[i,1]:
            current_dom[i,1] = x_i
            rightmost = True
            dims.append(i)
    if leftmost:
        return "min", dims, current_dom
    elif rightmost:
        return "max", dims, current_dom
    else:
        return "mid", dims, current_dom

# want to find hyperplane Ax=b
# A is given by nullspace of 
def points_to_plane(points):
    # must remove last point before taking null_space so we can find b
    last_point = points[-1,:]
    diff_mat = points[:-1,:]-last_point
    normal = null_space(diff_mat)
    # multiply by -1 so hyperplane is of form:
    # Ax + (-b) = 0
    const = -1*np.dot(last_point.reshape(1,d+1), normal)
    ineq = np.vstack([normal, const])
    
    if ineq[0] < 0:
        # flip all values to ensure region expressed by
        # inequality is convex of form ax+b >=0
        ineq = -1*ineq
        
    return ineq # ax+b >= 0

def point_plane_dist(point, plane):
    w = plane[:-1]
    b = plane[-1]
    dist = np.dot(point, w)+b
    dist = dist / np.linalg.norm(w)
    return dist

def update_to_midpoint(point, facet):
    # shifting a hyperplane to a parallel plane
    # is equivalent to simply updating its constant
    # the normal vector should remain the same
    new_facet = copy.copy(facet)
    
    # find perpendicular dist between pt and facet
    dist = point_plane_dist(point, facet)
    
    new_facet[-1] = new_facet[-1]-np.absolute(dist)/2
    
    return new_facet

def project_onto_plane(point, plane):
    distance = point_plane_dist(point, plane)
    
    normal = plane[:-1]
    unit_normal = normal/np.linalg.norm(normal)
    
    update_vec = -1*distance*unit_normal
    new_point = point+update_vec
    return new_point

def least_squares_update(point,plane,factor=1):
    new_point = copy.copy(point)
    
    y_plane = -1*(np.dot(plane[1:-1], point[1:])+plane[-1])/plane[0]
    
    new_point[0] = point[0]+factor*(y_plane-point[0])
    
    return new_point

def facet_from_subf_and_point(subfacet, point):
    if d == 1:
        slope = (subfacet[0]-point[0])/(subfacet[1]-point[1])
        b = subfacet[0]-slope*subfacet[1]
        facet = np.array([1,-1*slope, -1*b])
    else:
        raise NotImplementedError("todo.")
    return facet

def closest_violated_facet(violated_facets, new_point):
    # projection onto polytope is on the farthest violated facet plane
    for i in range(0, violated_facets.shape[0]):
        facet = violated_facets[i,:]
        dist = point_plane_dist(new_point, facet)
        if i == 0:
            max_dist = dist
            farthest_index = 0
        elif np.absolute(dist) > np.absolute(max_dist):
            # update farthest plane index
            max_dist = dist
            farthest_index = i
    return farthest_index

def project_to_intersection(point, planes):
    if planes.shape[0] == 1:
        return project_onto_plane(point, planes[0,:])
#     print("warning: the project to intersection is actually a compute intersection of planes")
    A = planes[:,:-1]
    b = -1*planes[:,-1] # of form Ax=b, planes in form Ax+c = 0, so b = -c
    new_point = np.dot(np.linalg.inv(A), b)
    return new_point

def point_slope_transform(ineq):
    y_magnitude = ineq[0]
    # check if vertical
    if y_magnitude == 0:
        ineq = ineq/ineq[1]
    else:
        ineq = ineq/y_magnitude
        ineq[1] = -1*ineq[1]
    ineq[2] = -1*ineq[2]
    return ineq

def closest_point(point, other_points):
    for i in range(0, other_points.shape[0]):
        pt = other_points[i,:]
        this_dist = np.absolute(np.linalg.norm(point-pt))
        if i == 0 or this_dist < min_dist:
            min_dist = copy.copy(this_dist)
            index = copy.copy(i)
    return index

def compute_midpoint(point1, point2):
    midpoint = copy.copy(point1)
    difference = point2 - point1
    midpoint = midpoint + 1/2*difference
    return midpoint

def find_k_farthest_facets_to_point(point, facets,k):
    for i in range(0,facets.shape[0]):
        facet = facets[i,:]
        dist = point_plane_dist(new_point, facet)
        if i == 0:
            relevant_facets = np.array([facet])
            dists = np.array([dist])
        elif dists.shape[0] < k:
            dists = np.vstack([dists,dist])
            relevant_facets = np.vstack([relevant_facets,facet])
        elif dist > np.min(dists):
            ind = np.argmin(dists)
            dists[ind] = dist
            relevant_facets[ind] = facet
    return relevant_facets

def determine_vertices_to_update(facet, vertices):
    # compute distances to hyperplane
    distances = np.dot(vertices, facet[:-1])
    distances = np.add(distances, facet[-1])
    
    dists = []
    indices = []
    
    for i in range(0,len(distances)):
        if distances[i] <= 0:
            dists.append(distances[i])
            indices.append(i)
    
    if len(indices) == d+1:
        relevant_vertices = np.array(indices)
        remaining_vertices = np.array([])
    else:
        indices = np.array(indices)
        order = np.argsort(dists)
        ordered_vertex_indices = indices[order.tolist()]

        relevant_vertices = np.array(ordered_vertex_indices[0:d]).astype("int")
        remaining_vertices = np.array(ordered_vertex_indices[d:]).astype("int")

    
    return relevant_vertices, remaining_vertices

def compute_intersection(planes):
    A = planes[:,:-1]
    b = -1*planes[:,-1] # of form Ax=b, planes in form Ax+c = 0, so b = -c
    new_point = np.dot(np.linalg.inv(A), b)
    return new_point
    
class Convex_Polytope:
    # note: facets in d-space are subfacets in (d+1)-space
    def __init__(self, points, dim):
        self.d = dim
        
        first_facet = points_to_plane(points)
        self.facets = first_facet.T

        if self.d == 1:
            # only two vertices needed
            min_point = points[np.argmin(points[:,1]),:]
            max_point = points[np.argmax(points[:,1]),:]

            self.vertices = np.array([min_point, max_point])
        else:
            # find convex hull of coplanar_pts on init_facet
            hull = ConvexHull(points)

            vertex_indices = hull.vertices
            self.vertices = points[vertex_indices, :]
        
        # force all types to be float, not int
        self.facets = self.facets.astype("double")
        self.vertices = self.vertices.astype("double")
        
        self.case_number = 2
    
    def max_affine_value(self,x):
        return np.max(-1*np.dot(self.facets[:,1:-1], x.T)-self.facets[:,-1], axis=0)
    # return indices of all vertices on facet
    def compute_mse(self, X,y):
        num_data = X.shape[0]
        mse = 0
        for i in range(0,num_data):
            yhat = self.max_affine_value(X[i,:])
            mse = mse + (y[i]-yhat)**2
            
        return mse/num_pts

    def vertices_on(self, facet):
        indices = []

        for i in range(0,self.vertices.shape[0]):
            vertex = self.vertices[i,:]
            # vertex is within epsilon distance of facet
            if np.absolute(np.dot(facet[:-1],vertex.T)+facet[-1]) < epsilon:
                indices.append(i)
        return indices
    
    # return indices of all facets belonging to vertex
    def facets_of(self, vertex, facet_number=-1):
        indices = []
        for i in range(0,self.facets.shape[0]):
            # optional: skip a given facet
            if i == facet_number:
                continue
            facet = self.facets[i,:]
            # vertex is within epsilon distance of facet
            if np.absolute(np.dot(facet[:-1],vertex)+facet[-1]) < epsilon:
                indices.append(i)
        return indices
    
    def case3_update(self, facet_number, new_point):
        self.case_number = 3
        
        closest_facet = self.facets[facet_number,:]
        
        # compute updated facet value
        updated_facet = update_to_midpoint(new_point, closest_facet)
        

#         # check if facet update causes certain facets to be irrelevant
#         # this can be done by checking if there exists more than
#         # (d+1) facets below new facet        
        relevant_vertex_indices, discarded_vertex_indices = determine_vertices_to_update(updated_facet, self.vertices)
        
        # update each relevant vertex
        for i in range(0, len(relevant_vertex_indices)):
            vertex = self.vertices[relevant_vertex_indices[i],:]
            
            # facets of this vertex not including updated one
            incident_facet_indices = self.facets_of(vertex, facet_number)
            
            if len(incident_facet_indices) > d:
                self.graph(point= new_point,display=True, xlim=(0.5,1))
                raise Exception("too many incident facets")
            # collect facets which define new vertex
            if len(incident_facet_indices) == 0:
                # when the vertex is an endpoint
                new_vertex = project_onto_plane(vertex, updated_facet)
            else:
                # when vertex defines intersection
                
                incident_facets = self.facets[incident_facet_indices,:]
                all_facets = np.vstack([incident_facets, updated_facet])
            
                # intersection of all facets is null_space
                new_vertex = project_to_intersection(vertex, all_facets)
                
                # check if new vertex maintains max-affine function
                # if so, we must remove vertex and compute new vertex
                for facet in incident_facets:
                    facet_vertices = self.vertices_on(facet)

            self.vertices[relevant_vertex_indices[i],:] = new_vertex
            
        # update facet    
        self.facets[facet_number,:] = updated_facet
        
        # find farthest violated vertex
        if discarded_vertex_indices.shape[0] == 0:
            return
        
        distances_from_new_vertex = np.absolute(np.linalg.norm(self.vertices[discarded_vertex_indices,:] - new_vertex, axis=1))
        farthest_discarded_vertex_index = discarded_vertex_indices[np.argmax(distances_from_new_vertex)]
        farthest_discarded_vert = self.vertices[farthest_discarded_vertex_index,:].flatten()
        
        # remove all discarded vertices (not including farthest)
        # and their corresponding facets
        discarded_facet_indices = []
        for i in range(0,discarded_vertex_indices.shape[0]):
            vertex_ind = discarded_vertex_indices[i]
            if vertex_ind == farthest_discarded_vertex_index:
                continue
            facet_indices = self.facets_of(self.vertices[vertex_ind,:])
            discarded_facet_indices.extend(facet_indices)
       
        self.vertices = np.delete(self.vertices, discarded_vertex_indices.tolist(), axis=0)
        self.facets = np.delete(self.facets, discarded_facet_indices, axis=0)
        
        # update farthest violated vertex to intersection of all facets
        # belonging to this vertex after removal of violated facets
        # and the current updated facet
        facets = np.vstack([self.facets[self.facets_of(farthest_discarded_vert),:], updated_facet])
        
        if facets.shape[0] <= d:
            updated_vertex = project_onto_plane(farthest_discarded_vert, updated_facet)
        else:
            updated_vertex = compute_intersection(facets)
        
        self.vertices = np.vstack([self.vertices, updated_vertex])
        
    def case4_update(self, facet_number, new_point, from_case5 = False):
        if not from_case5:
            self.case_number = 4
            
        violated_facet = self.facets[facet_number, :].flatten()
        
        incident_vertices = self.vertices_on(violated_facet)
        vertices_on_facet = self.vertices[incident_vertices,:]
        
#         if t == 205:
#             self.graph(display=True, facet_numbers = [facet_number], vertex_numbers= incident_vertices,point=new_point)
#             print("verts: ", self.vertices)
#             self.graph(display=True, point=new_point, xlim=(-0.46,-0.44), ylim=(0.15,0.25))

        # remove violated facet from polytope
        self.facets = np.delete(self.facets, facet_number, axis=0)
        
        if d == 1:
            subfacets = vertices_on_facet
        else:
            # find convex hull of points on facet in(d-1)-dim space
            hull = ConvexHull(vertices_on_facet)
            subfacets = hull.equations

                
        for subfacet in subfacets:
            redundant_facet = False
            if (np.absolute(subfacet-new_point) < epsilon).all():
                redundant_facet = True
            else:
                new_facet = facet_from_subf_and_point(subfacet, new_point)
            for facet in self.facets:
                if (np.absolute(facet-new_facet) < epsilon).all():
                    # remove all 
                    redundant_facet = True
            if not redundant_facet:
                self.facets = np.vstack([self.facets, new_facet])
        
        
        self.vertices = np.vstack([self.vertices, new_point])
    
    def case5_update(self, violated_facet_numbers, new_point,t):
        self.case_number = 5
       
        violated_facets = self.facets[violated_facet_numbers,:]
        
#         if t == 133:
#             self.graph(display=True, point=new_point)
#             print("violated_facets: ", violated_facets)
#             self.graph(display=True, point=new_point, facet_numbers=violated_facet_numbers.tolist())
        closest_violated_index = closest_violated_facet(violated_facets, new_point)
        
        closest_facet = violated_facets[closest_violated_index,:]
        
        d_farthest_violated_facets = find_k_farthest_facets_to_point(new_point, violated_facets, d+1)
        projection = project_to_intersection(new_point,d_farthest_violated_facets)
        new_vertex = compute_midpoint(projection, new_point)
        # keep track of closest facet
        closest_facet_number = violated_facet_numbers[closest_violated_index]

        # remove this facet from relevant facets and re-compute distances
        violated_facets = np.delete(violated_facets, closest_violated_index, axis=0)
        new_distances = np.dot(violated_facets[:,:-1], new_vertex)+violated_facets[:,-1]
        
        if (new_distances >= 0 ).all():
            print("== Case A")
            neighboring_vertices = []
            for facet in violated_facets:
                incident_vertices_indices = self.vertices_on(facet)
                incident_vertices = self.vertices[incident_vertices_indices,:]
                neighboring_index = closest_point(new_point, incident_vertices)
                neighboring_vertices.append(incident_vertices_indices[neighboring_index])
        
            self.case4_update(closest_facet_number, new_vertex, from_case5=True)
            
            self.vertices = np.delete(self.vertices, neighboring_vertices, axis=0)
        else:
#             print("== Case B")
            d_farthest_violated_facets = find_k_farthest_facets_to_point(new_point, violated_facets, d+1)
            # find projection onto violated facets not including closest
            new_vertex = project_to_intersection(new_point,d_farthest_violated_facets)
            
#             print(t,"new vert: ", new_vertex)
            on_vertex = False
            neighboring_vertices = []
            for facet in violated_facets:
                incident_vertices_indices = self.vertices_on(facet)
                incident_vertices = self.vertices[incident_vertices_indices,:]
                for vertex in incident_vertices:
                    if (np.absolute(vertex-new_vertex) < epsilon).all():
                        on_vertex = True
                neighboring_index = closest_point(new_point, incident_vertices)
                neighboring_vertices.append(incident_vertices_indices[neighboring_index])

#             if t == 133:
#                 print("new_vert: ", new_vertex)
# #                 print(self.vertices)
# #                 for vertex in neighboring_vertices:
# #                     if (np.absolute(vertex-new_vertex) < epsilon).all():
# #                         print("vertex is same as ", vertex)
#                 self.graph(display=True, facet_numbers=violated_facet_numbers.tolist(), point=new_vertex, xlim=(-0.52,-0.30), ylim=(0,0.5))
            if not on_vertex:
                self.case4_update(closest_facet_number, new_vertex, from_case5=True)

                self.vertices = np.delete(self.vertices, neighboring_vertices, axis=0)
#             if t == 133:
#                 self.graph(display=True, point=new_vertex, xlim=(-0.52,-0.30), ylim=(0,0.5))



  
    def graph(self, point=None, original_data=False, display=False, xlim = (min_val-0.1, max_val+0.1), ylim = [], facet_numbers=[], vertex_numbers=[] ):
        fig = plt.figure(figsize=(15,15))
        num_inequalities, num_dims = self.facets.shape
        if num_dims > 3:
            raise Exception("Unable to graph more than 2-dim data")
#         print("facets drawn: ")
        for row in range(0,num_inequalities):
            facet = self.facets[row,:]
            inequality = point_slope_transform(facet)
            if inequality[0] == 0:
                x_vals = np.multiply(inequality[2],np.ones([num_steps]))
                y_vals = y
            else:
                x_vals = X
                y_vals = np.multiply(inequality[1],x_vals)+inequality[2]
            if len(facet_numbers) == 0 or row in facet_numbers:
#                 print(facet)
                plt.plot(x_vals, y_vals,"--")

        if original_data:
            plt.plot(X,y, "*g")
        
            
        if len(vertex_numbers) == 0:
            plt.plot(self.vertices[:,1], self.vertices[:,0], "*r")
        else:
            vertices = self.vertices[vertex_numbers,:]
            print("drawn vertices: ", vertices)
            plt.plot(vertices[:,1], vertices[:,0], "*r")
        
        if point is not None:
            plt.plot(point[1], point[0], "*b")
            
        if xlim:
            plt.xlim(xlim)
        else:
            plt.xlim((-2.1,2.1))
        
        if ylim:
            plt.ylim(ylim)
        else:
            plt.ylim((-2.1,2.1))
        plt.title("Timestep: " + str(t) + " Case: "+str(self.case_number))
        plt.savefig("frames/"+str(t)+".png")
        if display == True:
            plt.show()
        else:
            plt.close()
    def verify(self):
        end_verts = []
        for i in range(0,self.vertices.shape[0]):
            vertex = self.vertices[i,:]
            if np.absolute(self.max_affine_value(vertex[1:])- vertex[1]) < epsilon:
                print("=== ERROR: vertex ", i, " does not lie on any hyperplane")
            if len(self.facets_of(vertex)) != d+1:
                end_verts.append(i)
        # check if endpoint lies on edge of        
        # if t == 21:
        #     self.graph(display=True, facet_numbers = np.arange(0,self.facets.shape[0]), vertex_numbers = np.arange(0,self.vertices.shape[0]))
        if len(end_verts) > d+1:
            print("=== ERROR: too many endpoints. vertices ", end_verts, " are detected as endpoints")
            self.graph(display=True, facet_numbers = np.arange(0,self.facets.shape[0]), vertex_numbers = np.arange(0,self.vertices.shape[0]))


# points begin as belonging to a less than d-dim space
full_rank = False

mse_vals = []
for t in range(0,num_pts):
    new_point = get_point(t) # retrieve point of form (y, x1, x2, .. xd)
    print("t: ", t, "point: ", new_point)
    # case 0: first data point
    if t == 0:
#         print("==== Case 0: First Point")
        init_points = new_point
        
        # domain pair for checking if new point in bounds
        # data point in bounds if it is not the min or max value in any dim
        # dx2 array where [:,0] is min vals, and [:,1] is max_vals
        x_new = np.array(new_point[1:])
        current_dom = np.array([np.array([x_new, x_new]) for i in range(0,d)])
        continue
        
    # case 1: all points up until this timestep belong to a less than d-dim space
    elif not full_rank:
        
        # update current domain with new points
        endpoint, dim, current_dom = check_endpoint(new_point[1:], current_dom)
        
        # check points to see if they are of rank d
        if is_degenerate(init_points, new_point):
#             print("==== Case 1: Not Enough Points to span d-dimensional space")
            init_points = np.vstack([init_points,new_point]) 
            continue
        else:
#             print("=== found initial full rank region!")
#             print("==== Case 2: Planar region and new non-planar data point")
            full_rank = True
            
            polytope = Convex_Polytope(init_points, d)
            
            # now, we proceed with standard updates for new_point

    # figure out if data point lies in domain
    endpoint, dim, new_dom = check_endpoint(new_point[1:], current_dom)
#     print("Domain region: ", endpoint)
    
    # compute distances to each hyperplane
    distances = np.dot(polytope.facets[:,:-1], new_point)
    distances = np.add(distances, polytope.facets[:,-1])
#     print("dists: ", distances)
      
    if endpoint == "mid":
        if np.count_nonzero(distances == 0) >= 1:
            continue
        if (distances > 0).all():
#             print("==== Case 3: Not an endpoint, inside polytope")

            closest_facet_number = np.argmin(np.absolute(distances))
            
            # update subfacets and vertices
            polytope.case3_update(closest_facet_number, new_point)
            
        elif (np.count_nonzero(distances < 0) == 1):
#             print("==== Case 4: Not an endpoint, inside convex combo of all but one hyperplane")
            # identify constraining hyperplane
            violated_facet_number = np.nonzero((distances < 0))[0]
            
            polytope.case4_update(violated_facet_number, new_point)       
        else:          
#             print("==== Case 5: Not an endpoint, outside convex combo")
            
            # find projection of point on violated constraints
            violated_facet_numbers = np.where(distances < 0)[0]
            
            polytope.case5_update(violated_facet_numbers, new_point,t)
    
#     print("facets: ", polytope.facets)
    if debug == True:
        polytope.graph(point=new_point)
        if num_vertices - num_facets != 1:
            raise Exception("mismatch vertices and facets")

            
    num_facets = polytope.facets.shape[0]
    num_vertices = polytope.vertices.shape[0]
    
    
    problem_facet = [ 1,      0.33266071, -0.05814729]
#     problem_facet =  [ 1,         0.33757211, -0.05883911]
    problem_facet = [ 1,         -0.01548991, -0.21804208]
    for i in range(0,polytope.facets.shape[0]):
        facet = polytope.facets[i,:]
        if (np.absolute(problem_facet - facet) < epsilon).all():
            print("+=++==== FOUND BAD FACET")
#     problem_vert = [-0.21503504,  0.82120409]
#     problem_vert = [ 2.11057362e-01, -4.50920695e-01]
#     problem_vert = [ 2.11057362e-01, -4.50920695e-01]
#     for i in range(0,polytope.vertices.shape[0]):
#         vert = polytope.vertices[i,:]
#         if (np.absolute(problem_vert-vert) < .001).all():
#             print("===== FOUND VERT ADDED AT ", t, "vert: ", vert, "error: ", np.absolute(problem_vert-vert))
      
    polytope.verify()
    print("t: ", t, "Case number: ", polytope.case_number)
    if t == 245:
        print(np.min(distances))
#     # TODO: implement cases for when new points do not lie in domain
#     # of all the current points
#     elif endpoint == "left":
#         closest_ind = np.argmin(np.absolute(distances))
#         # use closest hyperplane

    mse_vals.append(polytope.compute_mse(X,y))
        
    
# print("=== facets: ", polytope.facets)
