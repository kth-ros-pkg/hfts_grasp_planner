#! /usr/bin/python

import sys
import igraph
import pickle
import math
import rospy
from rtree import index
from std_msgs.msg import String
from threading import RLock, Thread
from gi.repository import Gtk, Gdk


class GUI(Gtk.Window):
    def __init__(self):
        super(GUI, self).__init__()
        self.drawing_lock = RLock()
        rospy.Subscriber('/goal_region_graph', String, self.ros_message_received)
        self.node_selection_publisher = rospy.Publisher('/node_select', String, queue_size=1)
        self.set_title("GTK Debug Drawer")
        self.connect_after('destroy', self.destroy)
        # eventBox = Gtk.EventBox()
        # self.imageView = Gtk.Image()
        # eventBox.add(self.imageView)
        self.darea = Gtk.DrawingArea()
        self.darea.add_events(Gdk.EventMask.BUTTON_PRESS_MASK)
        self.darea.connect('button_press_event', self.button_clicked)
        self.darea.connect('draw', self.expose)
        scroll_window = Gtk.ScrolledWindow()
        scroll_window.add(self.darea)
        self.add(scroll_window)
        self.show_all()
        self.desired_width = 1920
        self.nominal_width = 1920
        self.desired_height = 1000
        self.min_node_distance = 35
        self.node_size = 10
        self.margin = 20
        self._has_graph = False
        self.terminated = False
        self.max_temp = 4.5
        self.graph = None
        self.layout = None
        self.plot = None
        self._node_index = None
        self.darea.set_size_request(self.desired_width, self.desired_height)

    def destroy(self, unknow_arg):
        Gtk.main_quit()
        self.terminated = True

    def ros_message_received(self, msg):
        graph = pickle.loads(msg.data)
        self.update_graph(graph)

    def update_graph(self, graph):
        rospy.logdebug('update_graph called')
        with self.drawing_lock:
            self.graph = graph
            self.layout = graph.layout('rt', root=0)
            self._has_graph = True
            layer_sizes = self.compute_layer_sizes()
            max_nodes_per_layer = max(layer_sizes)
            needed_width = max_nodes_per_layer * self.min_node_distance
            if needed_width > self.nominal_width:
                self.desired_width = max(self.desired_width, needed_width)
            else:
                self.desired_width = self.nominal_width
            self.graph.vs['shape'] = ['circle' if x else 'rect' for x in self.graph.vs['isActive']]
            self.graph.vs['color'] = [self.compute_color(x) for x in self.graph.vs]
            for vidx in range(len(self.graph.vs)):
                label = '%.3f\nc:%.2f\nbc:%.2f' % (self.graph.vs[vidx]['temperature'],
                                                   self.graph.vs[vidx]['coverage'],
                                                   self.graph.vs[vidx]['branch_coverage'])
                self.graph.vs[vidx]['label'] = label
            self.graph.vs['label_dist'] = 2
            self.graph.vs['size'] = self.node_size
            self._node_index = None
            self.darea.queue_draw()

    def compute_color(self, vertex):
        # color = self.baseColor + vertex['temperature'] / self.maxTemp * (self.hotColor - self.baseColor)
        # color = color * 255
        # return map(int, color)
        color_code = int(vertex['temperature'] / self.max_temp * 255.0)
        return color_code

    def expose(self, widget, cairo_context):
        cairo_surface = cairo_context.get_target()
        if self._has_graph:
            with self.drawing_lock:
                self.darea.set_size_request(self.desired_width, self.desired_height)
                cairo_context.translate(0, 0)
                self.plot = igraph.plot(self.graph, target=cairo_surface,
                                        bbox=(0, 0, self.desired_width, self.desired_height),
                                        layout=self.layout,
                                        palette=igraph.drawing.colors.GradientPalette('black', 'orange'),
                                        margin=self.margin)
                self.plot.redraw(cairo_context)

    def compute_node_index(self):
        prop = index.Property()
        prop.dimension = 2
        self._node_index = index.Index(properties=prop)
        for vidx in range(len(self.graph.vs)):
            position = self.get_screen_coordinates(vidx)
            position += position
            self._node_index.insert(vidx, position)

    def get_screen_coordinates(self, vidx):
        x_layout = self.layout[vidx][0]
        y_layout = self.layout[vidx][1]
        ([x_min, y_min], [x_max, y_max]) = self.layout.boundaries()
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range == 0.0:
            x_range = 1.0
        if y_range == 0.0:
            y_range = 1.0
        x_screen = (x_layout - x_min) / x_range * (self.desired_width - 2 * self.margin) + self.margin
        y_screen = (y_layout - y_min) / y_range * (self.desired_height - 2 * self.margin) + self.margin
        position = [x_screen, y_screen]
        return position

    def get_node(self, x, y):
        if self._node_index is None:
            raise ValueError('Node index is None')
        query_pos = [x,y,x,y]
        nns = list(self._node_index.nearest(query_pos))
        vidx = nns[0]
        position = self.get_screen_coordinates(vidx)
        dist = math.sqrt(math.pow(position[0] - x, 2) + math.pow(position[1] - y, 2))
        print dist
        if dist < self.node_size:
            return vidx
        else:
            return None

    def button_clicked(self, widget, event):
        print 'you clicked me at ', event.x, ' ', event.y
        if self._node_index is None:
            self.compute_node_index()
        node = self.get_node(event.x, event.y)
        if node is not None:
            print self.graph.vs[node]['unique_label']
            self.node_selection_publisher.publish(self.graph.vs[node]['unique_label'])
        else:
            print 'No node clicked'
        self.darea.queue_draw()

    def compute_layer_sizes(self):
        layerSizes = [1]
        d = 1
        visited = len(self.graph.vs) * [False]
        self._compute_layer_sizes_rec(0, d, layerSizes, visited)
        return layerSizes

    def _compute_layer_sizes_rec(self, node_id, depth, layer_sizes, visited):
        if len(layer_sizes) < depth + 1:
            layer_sizes.append(0)
        layer_sizes[depth] += self.graph.outdegree(node_id)
        visited[node_id] = True
        for child in self.graph.neighbors(node_id):
            if not visited[child]:
                self._compute_layer_sizes_rec(child, depth + 1, layer_sizes, visited)


class GUIThread(Thread):
    def run(self):
        print 'Creating Window'
        self.gtkWindow = GUI()
        print 'Created Window'
        Gtk.main()

    def join(self):
        Gtk.main_quit()

def main2():
    rospy.init_node('FreeSpaceProximitySamplerVisualizerServer')
    Gdk.threads_init()
    app = GUI()
    Gtk.main()

def main():
    Gdk.threads_init()
    app = GUI()
    while not rospy.is_shutdown() and not app.terminated:
        Gtk.main_iteration()
        rospy.sleep(0.1)

if __name__ == "__main__":
    sys.exit(main2())
