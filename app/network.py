import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd
import random
from pprint import pprint
from .newton_bisection import findroot

# TODO: create the lookup tables from appendix C in ch. 2 instead of computing directly



# TODO: Create numpy docs for each function

        # self.G = ig.Graph(n=5, 
        #                   edges=[[0,1],[2,3],[3,1],[1,4]],
        #                   edge_attrs={
        #                       'flowrate': [0.0, 0.0, 0.0, 0.0]
        #                       'flowarea': [0.0, 0.0, 0.0, 0.0]
        #                       },
        #                   vertex_attrs={
        #                       'type': ['j', 'j', 'j', 'j', 'o'],
        #                       'depth': [0.0,0.0,0.0,0.0,0.0]
        #                       },
        #                   directed=True)

       

class SubcatchmentGraph:
    """General Graph Structure, Parent of the three subgraphs."""
    def __init__(self, file=None):
        super(SubcatchmentGraph, self).__init__()
        if file == None:
            self.G = ig.Graph(n=3,edges=[],directed=True,
                              vertex_attrs={
                                  'area': np.array([10000.0,10000.0,10000.0]),
                                  'width': np.array([100.0,100.0,100.0]),
                                  'slope': np.array([0.005,0.002,0.004]),
                                  'n': np.array([0.017,0.017,0.017]),
                                  'invert': np.array([0.0,0.016,0.035]),
                                  'x': np.array([100,200,200]),
                                  'y': np.array([100,100,0]),
                                  'z': np.array([0,0,0]),
                                  'depth': np.array([0.0,0.0,0.0])
                                  })
        else:
            # TODO: Make csv also include subcatchment edges
            data = pd.read_csv(f"data/{file}.csv")
            data = data[data["type"].str.contains("SUBCATCHMENT")]
            n = data.shape[0]
            # pprint(n)
            # pprint(data["type"])
            # pprint(data["x"].astype(float))
            # pprint(data)
            # pprint(data["type"].str.contains("OUTFALL").astype(int))

            # Needed to create edges
            edges = []
            # TODO: See above todo. will need this stuff for subcatchment edges
            # mapToID = []
            # i = 0
            # for _, row in data.iterrows():
            #     mapToID.append((row["id"],i))
            #     i = i+1
            #
            #
            #
            # # Creates the edges by translating the node id's in the csv into 0-indexed sewer nodes
            # for _, row in data.iterrows():
            #     # pprint(f"{index}, {row["id"]}")
            #     if row["outgoing"] != -1:
            #         id = row["id"]
            #         outgoing = row["outgoing"]
            #         for pair in mapToID:
            #             if pair[0] == id:
            #                 id = pair[1]
            #             if pair[0] == outgoing:
            #                 outgoing = pair[1]
            #         edges.append( (id, outgoing))

            self.hydraulicCoupling = np.array(data["outgoing"].astype(int))
            self.G = ig.Graph(n=n,edges=edges,directed=True,
                  vertex_attrs={
                      'coupledID': np.array(data["id"].astype(int)),
                      'invert': np.zeros(n),
                      'x': np.array(data["x"].astype(float)),
                      'y': np.array(data["y"].astype(float)),
                      'z': np.array(data["z"].astype(float)),
                      'area': np.array([10000.0,10000.0,10000.0]),
                      'width': np.array([100.0,100.0,100.0]),
                      'slope': np.array(data["slope"].astype(float)),
                      'n': np.array([0.017 for _ in range(n)]),
                      'depth': np.zeros(n),
                      })

    def update(self, t, dt, rainfall):
        """
        Updates the attributes of the network using the ode defined in "ode".

        Parameters:
        -----------
        t : float
            initial time
        dt : float
            time between initial time and desired end time
        rainfall : float
            average rainfall measured over the time [t,t+dt]

        """
        def ode(t, x):
            """
            Solves d_t = f - alpha * (d-ds)^5/3.

            Parameters:
            -----------
            t: time
            x : variable of ode.
            """
            y = np.zeros(self.G.vcount())
            outflow = np.zeros(self.G.vcount())
            incomingRunoff = np.zeros(self.G.vcount())
            for i in self.G.topological_sorting():
                # calculate incoming runoff, using top sorting to guarantee the previous runoffs are already computed
                inEdges = self.G.vs[i].in_edges()
                for e in inEdges:
                    incomingRunoff[i] += self.G.vs['depth'][e.source]

                # alpha in manning equation
                a = (self.G.vs['width'][i] * np.power(self.G.vs['slope'][i], 0.5)) / (self.G.vs['area'][i] * self.G.vs['n'][i])
                depth_above_invert = np.maximum(x[i] - self.G.vs['invert'][i], 0.0)
                # outgoingRunoff
                outflow[i] = a * np.power(depth_above_invert, 5/3)
                y[i] = rainfall + incomingRunoff[i] - outflow[i]
            # print(f"incomingRunoff: {incomingRunoff}")
            return y
    
        # NOTE: RK45 returns an iterator we need to use solve_ivp
        solution = sc.integrate.solve_ivp(
            ode, 
            (t, t + dt), 
            self.G.vs['depth'], 
            method='RK45'
        )
        self.G.vs['depth'] = solution.y[:, -1]
        return solution.y[:,-1]

    def visualize(self, times, depths, fileName=None):
        """
        Visualize depth over time for each subcatchment.
        
        Parameters:
        -----------
        times : 1-d list
            Array of time points
        depths : 2-d list
            List where each element is an array of depths at that time point
            Should have shape (n_timesteps, n_vertices)
        """
        depths_array = np.array(depths)
        
        plt.figure(figsize=(10, 6))
        
        for i in range(self.G.vcount()):
            plt.plot(times, depths_array[:, i], 
                    label=f'Subcatchment {i}', 
                    # marker='o', 
                    linewidth=2)
        
        plt.xlabel('Time (hours)', fontsize=12)
        plt.ylabel('Depth (m)', fontsize=12)
        plt.title('Subcatchment Depth vs Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if fileName == None:
            fileName = "test"
        plt.savefig(f"figures/{fileName}.png")


class StreetGraph:
    """Graph of Street portion of Hydraulic Network."""
    def __init__(self, file=None):
        super(StreetGraph, self).__init__()
        if file == None:
            self.G = ig.Graph(n=5,edges=[(0,1),(2,3),(3,1),(1,4)],directed=True,
                              vertex_attrs={
                                  'invert': np.array([0.0,0.016,0.035]),
                                  'x': np.array([100.0,100.0,200.0,200.0,0.0]),
                                  'y': np.array([100.0,0.0,100.0,0.0,0.0]),
                                  # z choice based on subcatchment slope, except for 4
                                  # 4 is ARBITRARY
                                  'z': np.array([0.5,0.0,0.6,0.4,-0.1]),
                                  'depth': np.array([0.0,0.0,0.0,0.0,0.0]),
                                  # 0 - junction
                                  # 1 - outfall
                                  'type': np.array([0,0,0,0,1])
                                  # 'subcatchmentCoupling': np.array([-1,-1,-1,-1,-1])
                                  })
            # calculate the lengths of each pipe
            for e in self.G.es:
                s = np.array([self.G.vs[e.source]['x'], self.G.vs[e.source]['y'], self.G.vs[e.source]['z']])
                d = np.array([self.G.vs[e.target]['x'], self.G.vs[e.target]['y'], self.G.vs[e.target]['z']])
                self.G.es[e.index]['length'] = np.linalg.norm(s - d)
            # calculate the slope of each pipe
            for e in self.G.es:
                slope = self.G.vs[e.source]['z'] - self.G.vs[e.target]['z']
                if slope < 0.0001:
                    print(f"WARNING: slope for edge {e} is too small.")
                self.G.es[e.index]['slope'] = self.G.vs[e.source]['z'] - self.G.vs[e.target]['z']
            # print(self.G.es['slope'])
            # TODO: add offset height calculations
            # Needs to be given a priori
            self.G.es['offsetHeight'] = [0.0 for _ in range(self.G.ecount())]

            # Geometry of Pipes (Circular in this case)
            self.G.es['diam'] = [0.5,0.5,0.8,1.0]
            # TODO: Decide if this should be stored (or computed) elsewhere
            self.G.es['areaFull'] = 0.25*np.pi*np.power(self.G.es['diam'],2)
            self.G.es['hydraulicRadiusFull'] = np.multiply(0.25,self.G.es['diam'])
            self.G.es['sectionFactorFull'] = self.G.es['areaFull']*np.power(self.G.es['hydraulicRadiusFull'],2/3)

            # self.G.es['flow'] = [0.0,0.0,0.0,0.0]
            self.G.es['flow'] = [0.1,0.2,0.1,0.1]

            self._steadyFlow(0,[0.1,0.1,0.1,0.1])

            # Create plot to test circular functions
            # TODO: Add plot generation for theta between 0 and pi
            # Calculate all functions
            for i in range(4):
                self.graphGeometry(i,file=f"circularPipeGeometry{i}")
        else:
            data = pd.read_csv(f"data/{file}.csv")
            data = data[data["type"].str.contains("STREET")]
            n = data.shape[0]
            # pprint(n)
            # pprint(data["type"])
            # pprint(data["x"].astype(float))
            # pprint(data)
            # pprint(data["type"].str.contains("OUTFALL").astype(int))

            # Needed to create edges
            edges = []
            mapToID = []
            i = 0
            for _, row in data.iterrows():
                mapToID.append((row["id"],i))
                i = i+1



            # Creates the edges by translating the node id's in the csv into 0-indexed sewer nodes
            for _, row in data.iterrows():
                # pprint(f"{index}, {row["id"]}")
                if row["outgoing"] != -1:
                    id = row["id"]
                    outgoing = row["outgoing"]
                    for pair in mapToID:
                        if pair[0] == id:
                            id = pair[1]
                        if pair[0] == outgoing:
                            outgoing = pair[1]
                    edges.append( (id, outgoing))



            self.G = ig.Graph(n=n,edges=edges,directed=True,
                  vertex_attrs={
                      'coupledID': np.array(data["id"].astype(int)),
                      'invert': np.zeros(n),
                      'x': np.array(data["x"].astype(float)),
                      'y': np.array(data["y"].astype(float)),
                      'z': np.array(data["z"].astype(float)),
                      'depth': np.zeros(n),
                      # 0 - junction
                      # 1 - outfall
                      'type': np.array(data["type"].str.contains("OUTFALL").astype(int))
                      # TODO: I need to decide if coupling occurs at this level or the level above
                      # 'subcatchmentCoupling': np.array([-1,-1,-1,-1,-1])
                      })
            # calculate the lengths of each pipe
            for e in self.G.es:
                s = np.array([self.G.vs[e.source]['x'], self.G.vs[e.source]['y'], self.G.vs[e.source]['z']])
                d = np.array([self.G.vs[e.target]['x'], self.G.vs[e.target]['y'], self.G.vs[e.target]['z']])
                self.G.es[e.index]['length'] = np.linalg.norm(s - d)
            # calculate the slope of each pipe
            for e in self.G.es:
                slope = self.G.vs[e.source]['z'] - self.G.vs[e.target]['z']
                if slope < 0.0001:
                    print(f"WARNING: slope for edge ({e.source}, {e.target}) is too small.")
                    print(f"{e.source}: ({self.G.vs[e.source]['x']}, {self.G.vs[e.source]['y']}, {self.G.vs[e.source]['z']})")
                    print(f"{e.target}: ({self.G.vs[e.target]['x']}, {self.G.vs[e.target]['y']}, {self.G.vs[e.target]['z']})")
                self.G.es[e.index]['slope'] = self.G.vs[e.source]['z'] - self.G.vs[e.target]['z']
            # pprint(f"Slopes: {self.G.es['slope']}")
            # pprint(f"Length: {self.G.es['length']}")
            # TODO: add offset height calculations
            # Needs to be given a priori
            self.G.es['offsetHeight'] = [0.0 for _ in range(self.G.ecount())]

            # Geometry of Pipes (Circular in this case)
            self.G.es['diam'] = [0.5 for _ in self.G.es]
            # pprint(f"Diam: {self.G.es['diam']}")
            # TODO: Decide if this should be stored (or computed) elsewhere
            self.G.es['areaFull'] = 0.25*np.pi*np.power(self.G.es['diam'],2)
            self.G.es['hydraulicRadiusFull'] = np.multiply(0.25,self.G.es['diam'])
            self.G.es['sectionFactorFull'] = self.G.es['areaFull']*np.power(self.G.es['hydraulicRadiusFull'],2/3)

            self.G.es['flow'] = np.zeros(self.G.ecount())
            # pprint(self.G.summary())

            # self._steadyFlow(0,[0.1,0.1,0.1,0.1])
        
    def update(self, t, dt, rainfall):
        """
        Updates the attributes of the network using the kinematic Model.

        Parameters:
        -----------
        t : float
            initial time
        dt : float
            time between initial time and desired end time
        rainfall : float
            average rainfall measured over the time [t,t+dt]

        Returns:
        --------
        depths : list
            Updated depths ordered by igraph id

        """
        def kineticFlow(t, x):
            """
            TODO: List assumptions. Uses mannings equation and continuity of mass to take the total inflow and write it as
            a discharge considering the pipe shape.

            Parameters:
            -----------
            t : float
                the current time in the ode
            x : list(float)
                list of depths
            """
            #1. check acyclic
            if not self.G.is_dag():
                raise ValueError("Street Network must be acyclic.")

            #2. top sort
            order = self.G.topological_sorting()

            for nid in order:
                #3. Get inflows
                # TODO: Continue
                pass
        pass


    def graphGeometry(self, id, file=None):
        theta = np.linspace(0.01, 2*np.pi - 0.01, 1000)
        area = [self._areaFromAngle(t)[id] for t in theta]
        d = [self._depth(t)[id] for t in theta]
        sf = [self._sectionFactor(t)[id] for t in theta]
        wp = [self._wettedPerimeter(t)[id] for t in theta]
        hr = [self._hydraulicRadius(t)[id] for t in theta]
        wp_deriv = [self._wettedPerimeterDerivative(t)[id] for t in theta]
        sf_deriv = [self._sectionFactorDerivative(t)[id] for t in theta]

        # Create subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Circular Pipe Functions vs Central Angle θ\n', 
                     fontsize=16, fontweight='bold')

        # Plot 1: Area
        axes[0, 0].plot(theta, area, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('θ (radians)')
        axes[0, 0].set_ylabel('Area (m²)')
        axes[0, 0].set_title('Cross-sectional Area')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=self.G.es['areaFull'][0], color='r', linestyle='--', alpha=0.5, label='Full Area')
        axes[0, 0].legend()

        # Plot 2: Depth
        axes[0, 1].plot(theta, d, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('θ (radians)')
        axes[0, 1].set_ylabel('Depth (m)')
        axes[0, 1].set_title('Flow Depth')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=self.G.es['diam'][id], color='r', linestyle='--', alpha=0.5, label='Full Depth')
        axes[0, 1].legend()

        # Plot 3: Section Factor
        axes[0, 2].plot(theta, sf, 'r-', linewidth=2)
        axes[0, 2].set_xlabel('θ (radians)')
        axes[0, 2].set_ylabel('Section Factor (m^(8/3))')
        axes[0, 2].set_title('Section Factor')
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Wetted Perimeter
        axes[1, 0].plot(theta, wp, 'c-', linewidth=2)
        axes[1, 0].set_xlabel('θ (radians)')
        axes[1, 0].set_ylabel('Wetted Perimeter (m)')
        axes[1, 0].set_title('Wetted Perimeter')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Hydraulic Radius
        axes[1, 1].plot(theta, hr, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('θ (radians)')
        axes[1, 1].set_ylabel('Hydraulic Radius (m)')
        axes[1, 1].set_title('Hydraulic Radius')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=self.G.es['hydraulicRadiusFull'][id], color='r', linestyle='--', alpha=0.5, label='Full')
        axes[1, 1].legend()

        # Plot 6: Wetted Perimeter Derivative
        axes[1, 2].plot(theta, wp_deriv, 'orange', linewidth=2)
        axes[1, 2].set_xlabel('θ (radians)')
        axes[1, 2].set_ylabel('dP/dθ')
        axes[1, 2].set_title('Wetted Perimeter Derivative')
        axes[1, 2].grid(True, alpha=0.3)

        # Plot 7: Section Factor Derivative
        axes[2, 0].plot(theta, sf_deriv, 'purple', linewidth=2)
        axes[2, 0].set_xlabel('θ (radians)')
        axes[2, 0].set_ylabel('dSF/dθ')
        axes[2, 0].set_title('Section Factor Derivative')
        axes[2, 0].grid(True, alpha=0.3)

        # Plot 8: Fill Ratio (Area/AreaFull)
        fill_ratio = area / self.G.es['areaFull'][id]
        axes[2, 1].plot(theta, fill_ratio, 'brown', linewidth=2)
        axes[2, 1].set_xlabel('θ (radians)')
        axes[2, 1].set_ylabel('Fill Ratio')
        axes[2, 1].set_title('Area Fill Ratio (A/A_full)')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_ylim([0, 1.1])

        # Plot 9: Depth vs Area (useful relationship)
        axes[2, 2].plot(area, d, 'navy', linewidth=2)
        axes[2, 2].set_xlabel('Area (m²)')
        axes[2, 2].set_ylabel('Depth (m)')
        axes[2, 2].set_title('Depth vs Area Relationship')
        axes[2, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        if file == None:
            file = 'streetPipeFunctions'
        plt.savefig(f'figures/{file}.png', dpi=300, bbox_inches='tight')

    def visualize(self, times, depths, fileName=None):
        """
        Visualize depth over time for each subcatchment.
        
        Parameters:
        -----------
        times : 1-d list
            Array of time points
        depths : 2-d list
            List where each element is an array of depths at that time point
            Should have shape (n_timesteps, n_vertices)
        """
        depths_array = np.array(depths)
        
        plt.figure(figsize=(10, 6))
        
        for i in range(self.G.vcount()):
            plt.plot(times, depths_array[:, i], 
                    label=f'Subcatchment {i}', 
                    # marker='o', 
                    linewidth=2)
        
        plt.xlabel('Time (hours)', fontsize=12)
        plt.ylabel('Depth (m)', fontsize=12)
        plt.title('Subcatchment Depth vs Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if fileName == None:
            fileName = "test"
        plt.savefig(f"figures/{fileName}.png")




class SewerGraph:
    """Graph of Sewer portion of Hydraulic Network."""
    def __init__(self, file=None):
        super(SewerGraph, self).__init__()
        if file == None:
            self.G = ig.Graph(n=5,edges=[(0,1),(2,3),(3,1),(1,4)],directed=True,
                              vertex_attrs={
                                  'invert': np.array([0.0,0.016,0.035]),
                                  'x': np.array([100.0,100.0,200.0,200.0,0.0]),
                                  'y': np.array([100.0,0.0,100.0,0.0,0.0]),
                                  # z choice based on subcatchment slope, except for 4
                                  # 4 is ARBITRARY
                                  'z': np.array([0.5,0.0,0.6,0.4,-0.1]),
                                  'depth': np.array([0.0,0.0,0.0,0.0,0.0]),
                                  # 0 - junction
                                  # 1 - outfall
                                  'type': np.array([0,0,0,0,1])
                                  # 'subcatchmentCoupling': np.array([-1,-1,-1,-1,-1])
                                  })
            # calculate the lengths of each pipe
            for e in self.G.es:
                s = np.array([self.G.vs[e.source]['x'], self.G.vs[e.source]['y'], self.G.vs[e.source]['z']])
                d = np.array([self.G.vs[e.target]['x'], self.G.vs[e.target]['y'], self.G.vs[e.target]['z']])
                self.G.es[e.index]['length'] = np.linalg.norm(s - d)
            # calculate the slope of each pipe
            for e in self.G.es:
                slope = self.G.vs[e.source]['z'] - self.G.vs[e.target]['z']
                if slope < 0.0001:
                    print(f"WARNING: slope for edge {e} is too small.")
                self.G.es[e.index]['slope'] = self.G.vs[e.source]['z'] - self.G.vs[e.target]['z']
            # print(self.G.es['slope'])
            # TODO: add offset height calculations
            # Needs to be given a priori
            self.G.es['offsetHeight'] = [0.0 for _ in range(self.G.ecount())]

            # Geometry of Pipes (Circular in this case)
            self.G.es['diam'] = [0.5,0.5,0.8,1.0]
            # TODO: Decide if this should be stored (or computed) elsewhere
            self.G.es['areaFull'] = 0.25*np.pi*np.power(self.G.es['diam'],2)
            self.G.es['hydraulicRadiusFull'] = np.multiply(0.25,self.G.es['diam'])
            self.G.es['sectionFactorFull'] = self.G.es['areaFull']*np.power(self.G.es['hydraulicRadiusFull'],2/3)

            # self.G.es['flow'] = [0.0,0.0,0.0,0.0]
            self.G.es['flow'] = [0.1,0.2,0.1,0.1]

            self._steadyFlow(0,[0.1,0.1,0.1,0.1])

            # Create plot to test circular functions
            # TODO: Add plot generation for theta between 0 and pi
            # Calculate all functions
            for i in range(4):
                self.graphGeometry(i,file=f"circularPipeGeometry{i}")
        else:
            data = pd.read_csv(f"data/{file}.csv")
            data = data[data["type"].str.contains("SEWER")]
            n = data.shape[0]
            # pprint(n)
            # pprint(data["type"])
            # pprint(data["x"].astype(float))
            # pprint(data)
            # pprint(data["type"].str.contains("OUTFALL").astype(int))

            # Needed to create edges
            edges = []
            mapToID = []
            i = 0
            for _, row in data.iterrows():
                mapToID.append((row["id"],i))
                i = i+1



            # Creates the edges by translating the node id's in the csv into 0-indexed sewer nodes
            for _, row in data.iterrows():
                # pprint(f"{index}, {row["id"]}")
                if row["outgoing"] != -1:
                    id = row["id"]
                    outgoing = row["outgoing"]
                    for pair in mapToID:
                        if pair[0] == id:
                            id = pair[1]
                        if pair[0] == outgoing:
                            outgoing = pair[1]
                    edges.append( (id, outgoing))



            self.G = ig.Graph(n=n,edges=edges,directed=True,
                  vertex_attrs={
                      'coupledID': np.array(data["id"].astype(int)),
                      'invert': np.zeros(n),
                      'x': np.array(data["x"].astype(float)),
                      'y': np.array(data["y"].astype(float)),
                      'z': np.array(data["z"].astype(float)),
                      'depth': np.zeros(n),
                      # 0 - junction
                      # 1 - outfall
                      'type': np.array(data["type"].str.contains("OUTFALL").astype(int))
                      # TODO: I need to decide if coupling occurs at this level or the level above
                      # 'subcatchmentCoupling': np.array([-1,-1,-1,-1,-1])
                      })
            # calculate the lengths of each pipe
            for e in self.G.es:
                s = np.array([self.G.vs[e.source]['x'], self.G.vs[e.source]['y'], self.G.vs[e.source]['z']])
                d = np.array([self.G.vs[e.target]['x'], self.G.vs[e.target]['y'], self.G.vs[e.target]['z']])
                self.G.es[e.index]['length'] = np.linalg.norm(s - d)
            # calculate the slope of each pipe
            for e in self.G.es:
                slope = self.G.vs[e.source]['z'] - self.G.vs[e.target]['z']
                if slope < 0.0001:
                    print(f"WARNING: slope for edge ({e.source}, {e.target}) is too small.")
                    print(f"{e.source}: ({self.G.vs[e.source]['x']}, {self.G.vs[e.source]['y']}, {self.G.vs[e.source]['z']})")
                    print(f"{e.target}: ({self.G.vs[e.target]['x']}, {self.G.vs[e.target]['y']}, {self.G.vs[e.target]['z']})")
                self.G.es[e.index]['slope'] = self.G.vs[e.source]['z'] - self.G.vs[e.target]['z']
            # pprint(f"Slopes: {self.G.es['slope']}")
            # pprint(f"Length: {self.G.es['length']}")
            # TODO: add offset height calculations
            # Needs to be given a priori
            self.G.es['offsetHeight'] = [0.0 for _ in range(self.G.ecount())]

            # Geometry of Pipes (Circular in this case)
            self.G.es['diam'] = [0.5 for _ in self.G.es]
            # pprint(f"Diam: {self.G.es['diam']}")
            # TODO: Decide if this should be stored (or computed) elsewhere
            self.G.es['areaFull'] = 0.25*np.pi*np.power(self.G.es['diam'],2)
            self.G.es['hydraulicRadiusFull'] = np.multiply(0.25,self.G.es['diam'])
            self.G.es['sectionFactorFull'] = self.G.es['areaFull']*np.power(self.G.es['hydraulicRadiusFull'],2/3)

            self.G.es['flow'] = np.zeros(self.G.ecount())
            # pprint(self.G.summary())

            # self._steadyFlow(0,[0.1,0.1,0.1,0.1])
        
    
    def _steadyFlow(self, t, x):
        """
        TODO: List assumptions. Uses mannings equation to take the total inflow and write it as
        a discharge considering the pipe shape.

        Parameters:
        -----------
        t : float
            the current time in the ode
        x : list(float)
            list of depths
        """
        #1. check acyclic
        if not self.G.is_dag():
            raise ValueError("Sewer Network must be acyclic.")

        #2. top sort
        order = self.G.topological_sorting()

        for nid in order:
            #3. Get inflows
            incomingEdges = self.G.incident(nid, mode='in')
            pprint(f"incoming: {incomingEdges}")
            inflow = np.sum([self.G.es['flow'][e] for e in incomingEdges])
            pprint(f"inflow: {inflow}")

    def _getDepthCircularPipe(self, q, diam, n, s):
        """
        Compute depth for given flow on Circular Pipe using Manning equation.
        Solves Q = (1/n) * A * R^(2/3) * S^(1/2) for depth.
        Using Newton-Raphson.
        """
        pass

    def _manning(self, area, hydraulicRadius, n, slope):
        """ Computes Manning from area, radius, n, slope."""
        if area <= 0 or hydraulicRadius <= 0 or slope <= 0:
            return 0.0

        return (1/n)* area * np.power(hydraulicRadius,2/3) * np.power(slope, 0.5)

    def _solveAverageDepth(self, flow):
        if flow <= 0:
            return 0.0
        if slope <= 0:
            slope = 0.0001

        # if flow >= self.G.es['']

    # TODO: Switch over to lookup tables
    def _angleFromArea(self, area):
        """computes the central angle by cross sectional flow area. A = A_{full} (theta - sin theta) / 2 pi"""
        a = np.divide(area, self.G.es['areaFull'])
        theta = 0.031715 - 12.79384 * a + 8.28479 * np.power(a,0.5)
        dtheta = 1e15
        while np.any(np.abs(dtheta) > 0.0001):
            denom = 1 - np.cos(theta)
            if np.any(denom == 0):
                raise ValueError("Division by zero: getAngleFromArea theta is 0")
            dtheta = 2 * np.pi * a - (theta - np.sin(theta)) / (1 - np.cos(theta))
            theta += dtheta
            # pprint(dtheta)

        return theta 

    def _areaFromAngle(self, theta):
        return np.multiply(self.G.es['areaFull'],(theta - np.sin(theta))) / (2*np.pi)

    def _depth(self, theta):
        depth = 0.5*np.multiply(self.G.es['diam'] , (1 - np.cos(theta / 2)))
        return depth

    def _sectionFactor(self, theta):
        return np.multiply(self.G.es['sectionFactorFull'] , np.power(theta - np.sin(theta), 5/3)) / (2*np.pi * np.power(theta,2/3))

    def _wettedPerimeter(self, theta):
        return 0.5 * np.multiply(theta , self.G.es['diam'])

    def _wettedPerimeterDerivative(self, theta):
        return np.divide(4, np.multiply(self.G.es['diam'],(1 - np.cos(theta))))

    def _hydraulicRadius(self, theta):
        return np.divide(self._areaFromAngle(theta), self._wettedPerimeter(theta))

    def _sectionFactorDerivative(self, theta):
        return ((5/3) - (2/3) * np.multiply(self._wettedPerimeterDerivative(theta) , self._hydraulicRadius(theta)))* np.power(self._hydraulicRadius(theta),2/3)

    # TODO: Add "Analytical Functions for Circular Cross Sections"
    def update(self, t, dt, rainfall):
        """
        Updates the attributes of the network using the ode defined in "ode".

        Parameters:
        -----------
        t : float
            initial time
        dt : float
            time between initial time and desired end time
        rainfall : float
            average rainfall measured over the time [t,t+dt]

        Returns:
        --------
        depths : list
            Updated depths ordered by igraph id

        """
        def ode(t, x):
            """
            Solves d_t = f - alpha * (d-ds)^5/3.

            Parameters:
            -----------
            t: time
            x : variable of ode.
            """
            y = np.zeros(self.G.vcount())
            outflow = np.zeros(self.G.vcount())
            incomingRunoff = np.zeros(self.G.vcount())
            for i in self.G.topological_sorting():
                # calculate incoming runoff, using top sorting to guarantee the previous runoffs are already computed
                inEdges = self.G.vs[i].in_edges()
                for e in inEdges:
                    incomingRunoff[i] += self.G.vs['depth'][e.source]

                # alpha in manning equation
                a = (self.G.vs['width'][i] * np.power(self.G.vs['slope'][i], 0.5)) / (self.G.vs['area'][i] * self.G.vs['n'][i])
                depth_above_invert = np.maximum(x[i] - self.G.vs['invert'][i], 0.0)
                # outgoingRunoff
                outflow[i] = a * np.power(depth_above_invert, 5/3)
                y[i] = rainfall + incomingRunoff[i] - outflow[i]
            print(f"incomingRunoff: {incomingRunoff}")
            return y
    
        # NOTE: RK45 returns an iterator we need to use solve_ivp
        solution = sc.integrate.solve_ivp(
            ode, 
            (t, t + dt), 
            self.G.vs['depth'], 
            method='RK45'
        )
        self.G.vs['depth'] = solution.y[:, -1]
        return solution.y[:,-1]

    def graphGeometry(self, id, file=None):
        theta = np.linspace(0.01, 2*np.pi - 0.01, 1000)
        area = [self._areaFromAngle(t)[id] for t in theta]
        d = [self._depth(t)[id] for t in theta]
        sf = [self._sectionFactor(t)[id] for t in theta]
        wp = [self._wettedPerimeter(t)[id] for t in theta]
        hr = [self._hydraulicRadius(t)[id] for t in theta]
        wp_deriv = [self._wettedPerimeterDerivative(t)[id] for t in theta]
        sf_deriv = [self._sectionFactorDerivative(t)[id] for t in theta]

        # Create subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Circular Pipe Functions vs Central Angle θ\n', 
                     fontsize=16, fontweight='bold')

        # Plot 1: Area
        axes[0, 0].plot(theta, area, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('θ (radians)')
        axes[0, 0].set_ylabel('Area (m²)')
        axes[0, 0].set_title('Cross-sectional Area')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=self.G.es['areaFull'][0], color='r', linestyle='--', alpha=0.5, label='Full Area')
        axes[0, 0].legend()

        # Plot 2: Depth
        axes[0, 1].plot(theta, d, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('θ (radians)')
        axes[0, 1].set_ylabel('Depth (m)')
        axes[0, 1].set_title('Flow Depth')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=self.G.es['diam'][id], color='r', linestyle='--', alpha=0.5, label='Full Depth')
        axes[0, 1].legend()

        # Plot 3: Section Factor
        axes[0, 2].plot(theta, sf, 'r-', linewidth=2)
        axes[0, 2].set_xlabel('θ (radians)')
        axes[0, 2].set_ylabel('Section Factor (m^(8/3))')
        axes[0, 2].set_title('Section Factor')
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Wetted Perimeter
        axes[1, 0].plot(theta, wp, 'c-', linewidth=2)
        axes[1, 0].set_xlabel('θ (radians)')
        axes[1, 0].set_ylabel('Wetted Perimeter (m)')
        axes[1, 0].set_title('Wetted Perimeter')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Hydraulic Radius
        axes[1, 1].plot(theta, hr, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('θ (radians)')
        axes[1, 1].set_ylabel('Hydraulic Radius (m)')
        axes[1, 1].set_title('Hydraulic Radius')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=self.G.es['hydraulicRadiusFull'][id], color='r', linestyle='--', alpha=0.5, label='Full')
        axes[1, 1].legend()

        # Plot 6: Wetted Perimeter Derivative
        axes[1, 2].plot(theta, wp_deriv, 'orange', linewidth=2)
        axes[1, 2].set_xlabel('θ (radians)')
        axes[1, 2].set_ylabel('dP/dθ')
        axes[1, 2].set_title('Wetted Perimeter Derivative')
        axes[1, 2].grid(True, alpha=0.3)

        # Plot 7: Section Factor Derivative
        axes[2, 0].plot(theta, sf_deriv, 'purple', linewidth=2)
        axes[2, 0].set_xlabel('θ (radians)')
        axes[2, 0].set_ylabel('dSF/dθ')
        axes[2, 0].set_title('Section Factor Derivative')
        axes[2, 0].grid(True, alpha=0.3)

        # Plot 8: Fill Ratio (Area/AreaFull)
        fill_ratio = area / self.G.es['areaFull'][id]
        axes[2, 1].plot(theta, fill_ratio, 'brown', linewidth=2)
        axes[2, 1].set_xlabel('θ (radians)')
        axes[2, 1].set_ylabel('Fill Ratio')
        axes[2, 1].set_title('Area Fill Ratio (A/A_full)')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_ylim([0, 1.1])

        # Plot 9: Depth vs Area (useful relationship)
        axes[2, 2].plot(area, d, 'navy', linewidth=2)
        axes[2, 2].set_xlabel('Area (m²)')
        axes[2, 2].set_ylabel('Depth (m)')
        axes[2, 2].set_title('Depth vs Area Relationship')
        axes[2, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        if file == None:
            file = 'circularPipeFunctions'
        plt.savefig(f'figures/{file}.png', dpi=300, bbox_inches='tight')

    def visualize(self, times, depths, fileName=None):
        """
        Visualize depth over time for each subcatchment.
        
        Parameters:
        -----------
        times : 1-d list
            Array of time points
        depths : 2-d list
            List where each element is an array of depths at that time point
            Should have shape (n_timesteps, n_vertices)
        """
        depths_array = np.array(depths)
        
        plt.figure(figsize=(10, 6))
        
        for i in range(self.G.vcount()):
            plt.plot(times, depths_array[:, i], 
                    label=f'Subcatchment {i}', 
                    # marker='o', 
                    linewidth=2)
        
        plt.xlabel('Time (hours)', fontsize=12)
        plt.ylabel('Depth (m)', fontsize=12)
        plt.title('Subcatchment Depth vs Time', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if fileName == None:
            fileName = "test"
        plt.savefig(f"figures/{fileName}.png")


    




if __name__ == "__main__":
    print("Dont call this directly :(")
    print("use one of the example files")

