"""Abstract graphs in ReGraph.

This module contains abstract classes for graph objects in ReGraph. Such
graph objects represent simple graphs with dictionary-like attributes
on nodes and edges.
"""
import json
import os

import warnings

from abc import ABC, abstractmethod

from regraph.exceptions import (ReGraphError,
                                GraphError,
                                GraphAttrsWarning,
                                )
from regraph.utils import (load_nodes_from_json,
                           load_edges_from_json,
                           generate_new_id,
                           normalize_attrs,
                           safe_deepcopy_dict,
                           set_attrs,
                           add_attrs,
                           remove_attrs,
                           merge_attributes,
                           keys_by_value,
                           )


class Graph(ABC):
    """Abstract class for graph objects in ReGraph."""

    @abstractmethod
    def nodes(self, data=False):
        """Return the list of nodes."""
        pass

    @abstractmethod
    def edges(self, data=False):
        """Return the list of edges."""
        pass

    @abstractmethod
    def get_node(self, n):
        """Get node attributes.

        Parameters
        ----------
        s : hashable
            Source node id.
        t : hashable,
            Target node id.
        """
        pass

    @abstractmethod
    def get_edge(self, s, t):
        """Get edge attributes.

        Parameters
        ----------
        s : hashable
            Source node id.
        t : hashable
            Target node id.
        """
        pass

    @abstractmethod
    def add_node(self, node_id, attrs=None):
        """Abstract method for adding a node.

        Parameters
        ----------
        node_id : hashable
            Prefix that is prepended to the new unique name.
        attrs : dict, optional
            Node attributes.
        """
        pass

    @abstractmethod
    def remove_node(self, node_id):
        """Remove node.

        Parameters
        ----------
        node_id : hashable
            Node to remove.
        """
        pass

    @abstractmethod
    def add_edge(self, s, t, attrs=None, **attr):
        """Add an edge to a graph.

        Parameters
        ----------
        s : hashable
            Source node id.
        t : hashable
            Target node id.
        attrs : dict
            Edge attributes.
        """
        pass

    @abstractmethod
    def remove_edge(self, source_id, target_id):
        """Remove edge from the graph.

        Parameters
        ----------
        s : hashable
            Source node id.
        t : hashable
            Target node id.
        """
        pass

    @abstractmethod
    def update_node_attrs(self, node_id, attrs, normalize=True):
        """Update attributes of a node.

        Parameters
        ----------
        node_id : hashable
            Node to update.
        attrs : dict
            New attributes to assign to the node
        """
        pass

    @abstractmethod
    def update_edge_attrs(self, s, t, attrs, normalize=True):
        """Update attributes of a node.

        Parameters
        ----------
        s : hashable
            Source node of the edge to update.
        t : hashable
            Target node of the edge to update.
        attrs : dict
            New attributes to assign to the node

        """
        pass

    @abstractmethod
    def successors(self, node_id):
        """Return the set of successors."""
        pass

    @abstractmethod
    def predecessors(self, node_id):
        """Return the set of predecessors."""
        pass

    @abstractmethod
    def find_matching(self, pattern, nodes=None):
        """Find matching of a pattern in a graph."""
        pass

    def print_graph(self):
        """Pretty-print the graph."""
        print("\nNodes:\n")
        for n in self.nodes():
            print(n, " : ", self.get_node(n))
        print("\nEdges:\n")
        for (n1, n2) in self.edges():
            print(n1, '->', n2, ' : ', self.get_edge(n1, n2))
        return

    def __str__(self):
        """String representation of the graph."""
        return "Graph({} nodes, {} edges)".format(
            len(self.nodes()), len(self.edges()))

    def __eq__(self, graph):
        """Eqaulity operator.

        Parameters
        ----------
        graph : regraph.Graph
            Another graph object


        Returns
        -------
        bool
            True if two graphs are equal, False otherwise.
        """
        if set(self.nodes()) != set(graph.nodes()):
            return False
        if set(self.edges()) != set(graph.edges()):
            return False
        for node in self.nodes():
            if self.get_node(node) != graph.get_node(node):
                return False
        for s, t in self.edges():
            if self.get_edge(s, t) != graph.get_edge(s, t):
                return False
        return True

    def __ne__(self, graph):
        """Non-equality operator."""
        return not (self == graph)

    def get_node_attrs(self, n):
        """Get node attributes.

        Parameters
        ----------
        n : hashable
            Node id.
        """
        return self.get_node(n)

    def get_edge_attrs(self, s, t):
        """Get edge attributes.

        Parameters
        ----------
        graph : networkx.(Di)Graph
        s : hashable, source node id.
        t : hashable, target node id.
        """
        return self.get_edge(s, t)

    def in_edges(self, node_id):
        """Return the set of in-coming edges."""
        return [(p, node_id) for p in self.predecessors(node_id)]

    def out_edges(self, node_id):
        """Return the set of out-going edges."""
        return [(node_id, s) for s in self.successors(node_id)]

    def add_nodes_from(self, node_list):
        """Add nodes from a node list.

        Parameters
        ----------
        node_list : iterable
            Iterable containing a collection of nodes, optionally,
            with their attributes
        """
        for n in node_list:
            if type(n) != str:
                try:
                    node_id, node_attrs = n
                    self.add_node(node_id, node_attrs)
                except (TypeError, ValueError):
                    self.add_node(n)
            else:
                self.add_node(n)

    def add_edges_from(self, edge_list):
        """Add edges from an edge list.

        Parameters
        ----------
        edge_list : iterable
            Iterable containing a collection of edges, optionally,
            with their attributes
        """
        for e in edge_list:
            if len(e) == 2:
                self.add_edge(e[0], e[1])
            elif len(e) == 3:
                self.add_edge(e[0], e[1], e[2])
            else:
                raise ReGraphError(
                    "Was expecting 2 or 3 elements per tuple, got %s." %
                    str(len(e))
                )

    def exists_edge(self, s, t):
        """Check if an edge exists.

        Parameters
        ----------
        s : hashable
            Source node id.
        t : hashable
            Target node id.
        """
        
        return((s, t) in self.edges())

    def set_node_attrs(self, node_id, attrs, normalize=True, update=True):
        """Set node attrs.

        Parameters
        ----------
        node_id : hashable
            Id of the node to update
        attrs : dict
            Dictionary with new attributes to set
        normalize : bool, optional
            Flag, when set to True attributes are normalized to be set-valued.
            True by default
        update : bool, optional
            Flag, when set to True attributes whose keys are not present
            in attrs are removed, True by default

        Raises
        ------
        GraphError
            If a node `node_id` does not exist.
        """
        if node_id not in self.nodes():
            raise GraphError("Node '{}' does not exist!".format(node_id))

        node_attrs = safe_deepcopy_dict(self.get_node(node_id))
        set_attrs(node_attrs, attrs, normalize, update)
        self.update_node_attrs(node_id, node_attrs, normalize)

    def add_node_attrs(self, node, attrs):
        """Add new attributes to a node.

        Parameters
        ----------
        node : hashable
            Id of a node to add attributes to.
        attrs : dict
            Attributes to add.

        Raises
        ------
        GraphError
            If a node `node_id` does not exist.
        """
        if node not in self.nodes():
            raise GraphError("Node '{}' does not exist!".format(node))

        node_attrs = safe_deepcopy_dict(self.get_node(node))
        add_attrs(node_attrs, attrs, normalize=True)
        self.update_node_attrs(node, node_attrs)

    def remove_node_attrs(self, node_id, attrs):
        """Remove attrs of a node specified by attrs_dict.

        Parameters
        ----------
        node_id : hashable
            Node whose attributes to remove.
        attrs : dict
            Dictionary with attributes to remove.

        Raises
        ------
        GraphError
            If a node with the specified id does not exist.
        """
        if node_id not in self.nodes():
            raise GraphError("Node '%s' does not exist!" % str(node_id))
        elif attrs is None:
            warnings.warn(
                "You want to remove attrs from '{}' with an empty attrs_dict!".format(
                    node_id), GraphAttrsWarning
            )

        node_attrs = safe_deepcopy_dict(self.get_node(node_id))
        remove_attrs(node_attrs, attrs, normalize=True)
        self.update_node_attrs(node_id, node_attrs)

    def set_edge_attrs(self, s, t, attrs, normalize=True, update=True):
        """Set edge attrs.

        Parameters
        ----------
        attrs : dict
            Dictionary with new attributes to set
        normalize : bool, optional
            Flag, when set to True attributes are normalized to be set-valued.
            True by default
        update : bool, optional
            Flag, when set to True attributes whose keys are not present
            in attrs are removed, True by default

        Raises
        ------
        GraphError
            If an edge between `s` and `t` does not exist.
        """
        if not self.exists_edge(s, t):
            raise GraphError(
                "Edge {}->{} does not exist".format(s, t))

        edge_attrs = safe_deepcopy_dict(self.get_edge(s, t))
        set_attrs(edge_attrs, attrs, normalize, update)
        self.update_edge_attrs(s, t, edge_attrs, normalize=normalize)

    def set_edge(self, s, t, attrs, normalize=True, update=True):
        """Set edge attrs.

        Parameters
        ----------
        s : hashable
            Source node id.
        t : hashable
            Target node id.
        attrs : dictionary
            Dictionary with attributes to set.

        Raises
        ------
        GraphError
            If an edge between `s` and `t` does not exist.
        """
        self.set_edge_attrs(s, t, attrs, normalize, update)

    def add_edge_attrs(self, s, t, attrs):
        """Add attributes of an edge in a graph.

        Parameters
        ----------
        s : hashable
            Source node id.
        t : hashable
            Target node id.
        attrs : dict
            Dictionary with attributes to remove.

        Raises
        ------
        GraphError
            If an edge between `s` and `t` does not exist.
        """
        if not self.exists_edge(s, t):
            raise GraphError(
                "Edge {}->{} does not exist".format(s, t))

        edge_attrs = safe_deepcopy_dict(self.get_edge(s, t))
        add_attrs(edge_attrs, attrs, normalize=True)
        self.update_edge_attrs(s, t, edge_attrs)

    def remove_edge_attrs(self, s, t, attrs):
        """Remove attrs of an edge specified by attrs.

        Parameters
        ----------
        s : hashable
            Source node id.
        t : hashable
            Target node id.
        attrs : dict
            Dictionary with attributes to remove.

        Raises
        ------
        GraphError
            If an edge between `s` and `t` does not exist.
        """
        if not self.exists_edge(s, t):
            raise GraphError(
                "Edge {}->{} does not exist".format(s, t))

        edge_attrs = safe_deepcopy_dict(self.get_edge(s, t))
        remove_attrs(edge_attrs, attrs, normalize=True)
        self.update_edge_attrs(s, t, edge_attrs)

    def clone_node(self, node_id, name=None):
        """Clone node.

        Create a new node, a copy of a node with `node_id`, and reconnect it
        with all the adjacent nodes of `node_id`.

        Parameters
        ----------
        node_id : hashable,
            Id of a node to clone.
        name : hashable, optional
            Id for the clone, if is not specified, new id will be generated.

        Returns
        -------
        new_node : hashable
            Id of the new node corresponding to the clone

        Raises
        ------
        GraphError
            If node wiht `node_id` does not exists or a node with
            `name` (clone's name) already exists.

        """
        if node_id not in self.nodes():
            raise GraphError("Node '{}' does not exist!".format(node_id))

        # generate new name for a clone
        if name is None:
            i = 1
            new_node = str(node_id) + str(i)
            while new_node in self.nodes():
                i += 1
                new_node = str(node_id) + str(i)
        else:
            if name in self.nodes():
                raise GraphError("Node '{}' already exists!".format(name))
            else:
                new_node = name

        self.add_node(new_node, self.get_node(node_id))

        # Connect all the edges
        self.add_edges_from(
            set([(n, new_node) for n, _ in self.in_edges(node_id)
                 if (n, new_node) not in self.edges()]))
        self.add_edges_from(
            set([(new_node, n) for _, n in self.out_edges(node_id)
                 if (new_node, n) not in self.edges()]))

        # Copy the attributes of the edges
        for s, t in self.in_edges(node_id):
            self.set_edge(
                s, new_node,
                safe_deepcopy_dict(self.get_edge(s, t)))
        for s, t in self.out_edges(node_id):
            self.set_edge(
                new_node, t,
                safe_deepcopy_dict(self.get_edge(s, t)))
        return new_node

    def relabel_node(self, node_id, new_id):
        """Relabel a node in the graph.

        Parameters
        ----------
        node_id : hashable
            Id of the node to relabel.
        new_id : hashable
            New label of a node.
        """
        if new_id in self.nodes():
            raise ReGraphError(
                "Cannot relabel '{}' to '{}', '{}' ".format(
                    node_id, new_id, new_id) +
                "already exists in the graph")
        self.clone_node(node_id, new_id)
        self.remove_node(node_id)

    def merge_nodes(self, nodes, node_id=None):
        """Merge a list of nodes.
        
        Parameters
        ----------
        nodes : iterable
            Collection of node ids to merge.
        node_id : hashable, optional
            Id of the new node corresponding to the result of merge.
        method : str, optional
            Method of node attributes merge, unused in simplified version.
        edge_method : str, optional
            Method of edge attributes merge, unused in simplified version.
            
        Returns
        -------
        node_id : hashable
            Id of the merged node
        
        Notes
        -----
        This simplified version specifically handles 't' and 'phase' attributes.
        't' values must be consistent across all merged nodes, while 'phase' values are added.
        """
        assert len(nodes) > 1, "More than one node should be specified for merging!"
        

        # Generate name for new node if not provided
        if node_id is None:
            node_id = "_".join(sorted([str(n) for n in nodes]))
            #if node_id in self.nodes():
            #    node_id = self.generate_new_node_id(node_id)
        elif node_id in self.nodes() and (node_id not in nodes):
            raise GraphError(
                f"New name for merged node is not valid: node with name '{node_id}' already exists!"
            )

        # Process node attributes
        merged_attrs = {}
        t_value = None
        phase_sum = 0
        
        # Check and collect attributes from all nodes
        merged_attrs["previous_ids"] = []
        actual_nodes = []
        for node in nodes:
            # Update nodes list with to match the current nodes in the graph
            # One of the nodes might be merged in the previous rounds
            try:
                attrs = self.get_node(node)
                actual_nodes.append(node)
            except IndexError:
                node, attrs = self.get_node_based_on_previous_ids(node)
                actual_nodes.append(node)
            # Handle 't' attribute - must be consistent
            if 't' in attrs:
                if t_value is None:
                    t_value = attrs['t']
                elif t_value != attrs['t']:
                    warnings.warn(f"Inconsistent 't' values found when merging nodes. Using t={t_value}")
            
            # Handle 'phase' attribute - values are added
            if 'phase' in attrs:
                phase_sum += attrs['phase']
                
            if 'graph_id' in attrs:
                merged_attrs['graph_id'] = attrs['graph_id']
            
            if 'previous_ids' in attrs:
                merged_attrs['previous_ids'].extend(attrs['previous_ids'])
            
            if 'id' in attrs:
                merged_attrs["previous_ids"].append(attrs["id"])
        
        # Set the merged attributes
        merged_attrs['t'] = t_value
        merged_attrs['phase'] = phase_sum

        # Collect source and target nodes (nodes connected to the merged nodes)
        source_nodes = set()  # nodes with edges coming into the merged nodes
        target_nodes = set()  # nodes with edges going from the merged nodes
        
        # Process each node to merge
        for node in actual_nodes:
            # Get incoming and outgoing edges
            in_edges = self.in_edges(node)
            out_edges = self.out_edges(node)
            
            # Collect source nodes (replacing merged nodes with the new node_id)
            source_nodes.update([s if s not in actual_nodes else node_id for s, _ in in_edges])

            # Collect target nodes (replacing merged nodes with the new node_id)
            target_nodes.update([t if t not in actual_nodes else node_id for _, t in out_edges])

            # Remove the processed node
            self.remove_node(node)
        
        # Create the new merged node with the collected attributes
        self.add_node(node_id, merged_attrs)
        
        # Create edges to/from the merged node
        for source in source_nodes:
            if source != node_id:  # Skip if it's a reference to the merged node itself
                self.add_edge(source, node_id)
        
        for target in target_nodes:
            if target != node_id:  # Skip if it's a reference to the merged node itself
                self.add_edge(node_id, target)
        
        return node_id          

    def copy_node(self, node_id, copy_id=None):
        """Copy node.

        Create a copy of a node in a graph. A new id for the copy is
        generated by regraph.primitives.unique_node_id.

        Parameters
        ----------
        node_id : hashable
            Node to copy.

        Returns
        -------
        new_name
            Id of the copy node.

        """
        if copy_id is None:
            copy_id = self.generate_new_node_id(node_id)
        if copy_id in self.nodes():
            raise ReGraphError(
                "Cannot create a copy of '{}' with id '{}', ".format(
                    node_id, copy_id) +
                "node '{}' already exists in the graph".format(copy_id))
        attrs = self.get_node(node_id)
        self.add_node(copy_id, attrs)
        return copy_id

    def relabel_nodes(self, mapping):
        """Relabel graph nodes inplace given a mapping.

        Similar to networkx.relabel.relabel_nodes:
        https://networkx.github.io/documentation/development/_modules/networkx/relabel.html

        Parameters
        ----------
        mapping: dict
            A dictionary with keys being old node ids and their values
            being new id's of the respective nodes.

        Raises
        ------
        ReGraphError
            If new id's do not define a set of distinct node id's.

        """
        unique_names = set(mapping.values())
        if len(unique_names) != len(self.nodes()):
            raise ReGraphError(
                "Attempt to relabel nodes failed: the IDs are not unique!")

        temp_names = {}
        # Relabeling of the nodes: if at some point new ID conflicts
        # with already existing ID - assign temp ID
        for key, value in mapping.items():
            if key != value:
                if value not in self.nodes():
                    new_name = value
                else:
                    new_name = self.generate_new_node_id(value)
                    temp_names[new_name] = value
                self.relabel_node(key, new_name)
        # Relabeling the nodes with the temp ID to their new IDs
        for key, value in temp_names.items():
            if key != value:
                self.relabel_node(key, value)
        return

    def generate_new_node_id(self, basename):
        """Generate new unique node identifier."""
        return generate_new_id(self.nodes(), basename)

    def filter_edges_by_attributes(self, attr_key, attr_cond):
        """Filter graph edges by attributes.

        Removes all the edges of the graph (inplace) that do not
        satisfy `attr_cond`.

        Parameters
        ----------
        attrs_key : hashable
            Attribute key
        attrs_cond : callable
            Condition for an attribute to satisfy: callable that returns
            `True` if condition is satisfied, `False` otherwise.

        """
        for (s, t) in self.edges():
            edge_attrs = self.get_edge(s, t)
            if (attr_key not in edge_attrs.keys() or
                    not attr_cond(edge_attrs[attr_key])):
                self.remove_edge(s, t)

    def to_json(self):
        """Create a JSON representation of a graph."""
        j_data = {"edges": [], "nodes": []}
        # dump nodes
        for node in self.nodes():
            node_data = {}
            node_data["id"] = node
            node_attrs = self.get_node(node)
            if node_attrs is not None:
                attrs = {}
                for key, value in node_attrs.items():
                    attrs[key] = value.to_json()
                node_data["attrs"] = attrs
            j_data["nodes"].append(node_data)

        # dump edges
        for s, t in self.edges():
            edge_data = {}
            edge_data["from"] = s
            edge_data["to"] = t
            edge_attrs = self.get_edge(s, t)
            if edge_attrs is not None:
                attrs = {}
                for key, value in edge_attrs.items():
                    attrs[key] = value.to_json()
                edge_data["attrs"] = attrs
            j_data["edges"].append(edge_data)
        return j_data

    def to_d3_json(self,
                   attrs=True,
                   node_attrs_to_attach=None,
                   edge_attrs_to_attach=None,
                   nodes=None):
        """Create a JSON representation of a graph."""
        j_data = {"links": [], "nodes": []}
        if nodes is None:
            nodes = self.nodes()
        # dump nodes
        for node in nodes:
            node_data = {}
            node_data["id"] = node
            if attrs:
                node_attrs = self.get_node(node)
                normalize_attrs(node_attrs)
                attrs_json = dict()
                for key, value in node_attrs.items():
                    attrs_json[key] = value.to_json()
                node_data["attrs"] = attrs_json
            else:
                node_attrs = self.get_node(node)
                if node_attrs_to_attach is not None:
                    for key in node_attrs_to_attach:
                        if key in node_attrs.keys():
                            node_data[key] = list(node_attrs[key])
            j_data["nodes"].append(node_data)

        # dump edges
        for s, t in self.edges():
            if s in nodes and t in nodes:
                edge_data = {}
                edge_data["source"] = s
                edge_data["target"] = t
                if attrs:
                    edge_attrs = self.get_edge(s, t)
                    normalize_attrs(edge_attrs)
                    attrs_json = dict()
                    for key, value in edge_attrs.items():
                        attrs_json[key] = value.to_json()
                    edge_data["attrs"] = attrs_json
                else:
                    if edge_attrs_to_attach is not None:
                        for key in edge_attrs_to_attach:
                            edge_attrs = self.get_edge(s, t)
                            if key in edge_attrs.keys():
                                edge_data[key] = list(edge_attrs[key])
                j_data["links"].append(edge_data)

        return j_data

    def export(self, filename):
        """Export graph to JSON file.

        Parameters
        ----------
        filename : str
            Name of the file to save the json serialization of the graph


        """
        with open(filename, 'w') as f:
            j_data = self.to_json()
            json.dump(j_data, f)
        return

    @classmethod
    def from_json(cls, json_data):
        """Create a NetworkX graph from a json-like dictionary.

        Parameters
        ----------
        json_data : dict
            JSON-like dictionary with graph representation
        """
        graph = cls()
        graph.add_nodes_from(load_nodes_from_json(json_data))
        graph.add_edges_from(load_edges_from_json(json_data))
        return graph

    @classmethod
    def load(cls, filename):
        """Load a graph from a JSON file.

        Create a `networkx.(Di)Graph` object from
        a JSON representation stored in a file.

        Parameters
        ----------
        filename : str
            Name of the file to load the json serialization of the graph


        Returns
        -------
        Graph object

        Raises
        ------
        ReGraphError
            If was not able to load the file

        """
        if os.path.isfile(filename):
            with open(filename, "r+") as f:
                j_data = json.loads(f.read())
                return cls.from_json(j_data)
        else:
            raise ReGraphError(
                "Error loading graph: file '{}' does not exist!".format(
                    filename)
            )

    def rewrite(self, rule, instance=None):
        """Perform SqPO rewiting of the graph with a rule.

        Parameters
        ----------
        rule : regraph.Rule
            SqPO rewriting rule
        instance : dict, optional
            Instance of the input rule. If not specified,
            the identity map of the rule's left-hand side
            is used

        """
        if instance is None:
            instance = {
                n: n for n in self.lhs.nodes()
            }

        # Restrictive phase
        p_g = dict()
        cloned_lhs_nodes = set()

        # Clone nodes
        for lhs, p_nodes in rule.cloned_nodes().items():
            for i, p in enumerate(p_nodes):
                if i == 0:
                    p_g[p] = instance[lhs]
                    cloned_lhs_nodes.add(lhs)
                else:
                    clone_id = self.clone_node(instance[lhs])
                    p_g[p] = clone_id

        # Delete nodes and add preserved nodes to p_g dictionary
        removed_nodes = rule.removed_nodes()
        for n in rule.lhs.nodes():
            if n in removed_nodes:
                self.remove_node(instance[n])
            elif n not in cloned_lhs_nodes:
                p_g[keys_by_value(rule.p_lhs, n)[0]] =\
                    instance[n]

        # Delete edges
        for u, v in rule.removed_edges():
            self.remove_edge(p_g[u], p_g[v])

        # Remove node attributes
        for p_node, attrs in rule.removed_node_attrs().items():
            self.remove_node_attrs(p_g[p_node], attrs)

        # Remove edge attributes
        for (u, v), attrs in rule.removed_edge_attrs().items():
            self.remove_edge_attrs(p_g[u], p_g[v], attrs)

        # Expansive phase
        rhs_g = dict()
        merged_nodes = set()

        # Merge nodes
        for rhs, p_nodes in rule.merged_nodes().items():
            merge_id = self.merge_nodes([p_g[p] for p in p_nodes])
            merged_nodes.add(rhs)
            rhs_g[rhs] = merge_id

        # Add nodes and add preserved nodes to the rhs_g dictionary
        added_nodes = rule.added_nodes()
        for n in rule.rhs.nodes():
            if n in added_nodes:
                if n in self.nodes():
                    new_id = self.generate_new_node_id(n)
                else:
                    new_id = n
                new_id = self.add_node(new_id)
                rhs_g[n] = new_id
            elif n not in merged_nodes:
                rhs_g[n] = p_g[keys_by_value(rule.p_rhs, n)[0]]

        # Add edges
        for u, v in rule.added_edges():
            if (rhs_g[u], rhs_g[v]) not in self.edges():
                self.add_edge(rhs_g[u], rhs_g[v])

        # Add node attributes
        #for rhs_node, attrs in rule.added_node_attrs().items():
        #    self.add_node_attrs(rhs_g[rhs_node], attrs)

        # Add edge attributes
        #for (u, v), attrs in rule.added_edge_attrs().items():
        #    self.add_edge_attrs(rhs_g[u], rhs_g[v], attrs)
            
        return rhs_g

    def number_of_edges(self, u, v):
        """Return number of directed edges from u to v."""
        return 1

    def ancestors(self, t):
        """Return the set of ancestors."""
        current_level = set(self.predecessors(t))
        visited = set()

        while len(current_level) > 0:
            next_level = set()
            for el in current_level:
                if el not in visited:
                    visited.add(el)
                    next_level.update([
                        p
                        for p in self.predecessors(el)
                        if p not in visited])
            current_level = next_level
        return visited

    def descendants(self, s):
        """Return the set of ancestors."""
        current_level = set(self.successors(s))
        visited = set()

        while len(current_level) > 0:
            next_level = set()
            for el in current_level:
                if el not in visited:
                    visited.add(el)
                    next_level.update([
                        p
                        for p in self.successors(el)
                        if p not in visited])
            current_level = next_level
        return visited
