from . import generic


def add_node(node_label, attrs=None, node_id=None):
    """Add a node to the Neo4j graph database.
    
    Creates a node with the given label and attributes. If no node_id is provided,
    Neo4j will generate a unique ID internally. Returns the unique ID of the created node.
    
    Parameters
    ----------
    node_label : str
        Label to assign to the node (e.g., 'Person', 'Product')
    attrs : dict, optional
        Dictionary of attributes to assign to the node
    node_id : str, optional
        Custom ID to assign to the node. If None, a unique ID will be generated
        
    Returns
    -------
    str
        The unique ID of the created node
    """
    # Initialize query with node creation
    query = f"CREATE (n:{node_label}"
    
    # Handle attributes
    if attrs is None:
        attrs = {}
    
    # Add the node_id to attributes if provided
    if node_id is not None:
        attrs['id'] = node_id
    
    # Format attributes for Cypher
    if attrs:
        attr_strings = []
        for key, value in attrs.items():
            # Handle different types of values
            if isinstance(value, str):
                value_str = f"'{value}'"
            elif isinstance(value, (list, set)):
                elements = [f"'{e}'" if isinstance(e, str) else str(e) for e in value]
                value_str = f"[{', '.join(elements)}]"
            else:
                value_str = str(value)
            attr_strings.append(f"{key}: {value_str}")
        
        query += " {" + ", ".join(attr_strings) + "}"
    
    # Complete the query to return the ID
    query += ")\n"
    query += "RETURN n.id AS node_id"
    
    return query


def add_edge(edge_var, source_var, target_var,
             edge_label, attrs=None, merge=False):
    """Generate query for edge creation.

    source_var
        Name of the variable corresponding to the source
        node
    target_var
        Name of the variable corresponding to the target
        node
    edge_label : optional
        Labels associated with the new edge
    attrs : dict, optional
        Attributes of the new edge
    """
    if merge:
        keyword = "MERGE"
    else:
        keyword = "CREATE"

    query = "{} ({})-[{}:{} {{ {} }}]->({})\n".format(
        keyword, source_var, edge_var, edge_label,
        generic.generate_attributes(attrs), target_var)
    return query


def remove_node(node_var):
    """Query for removal of a node (with side-effects)."""
    return "DETACH DELETE {}\n".format(node_var)


def remove_edge(edge_var, breakline=True):
    """Query for deleting an edge corresponding to the input variable.

    Parameters
    ----------
    edge_var
        Name of the variable corresponding to the edge to remove
    """
    return generic.delete_var(edge_var, False, breakline)


def remove_nodes(var_names, breakline=True):
    """Query for deleting nodes corresponding to the input variables.

    Parameters
    ----------
    var_names : iterable
        Collection of variables corresponding to nodes to remove
    """
    n = ""
    if breakline is True:
        n = "\n"

    return "DETACH DELETE {}{}".format(', '.join(v for v in var_names), n)


def find_matching(pattern, node_label, edge_label,
                  nodes=None, pattern_typing=None, undirected_edges=None):
    """Query that performs pattern match in the graph.

    Parameters
    ----------
    pattern : nx.(Di)Graph
        Graph object representing a pattern to search for
    nodes : iterable, optional
        Collection of ids of nodes to constraint the search space of matching
    node_label
        Label of the node to match, default is 'node'
    edge_label
        Label of the edges to match, default is 'edge'
    """
    if undirected_edges is None:
        undirected_edges = []
    # normalize pattern typing
    pattern_nodes = list(pattern.nodes())
    pattern_edges = list(pattern.edges(data=True))

    query =\
        "MATCH {}".format(
            ", ".join(
                "({}:{})".format(n, node_label)
                for n in pattern_nodes))
    if len(pattern.edges()) > 0:
        query += ", {}".format(
            ", ".join("({})-[{}_to_{}:{}]-{}({})".format(
                u, u, v, edge_label,
                "" if (u, v) in undirected_edges else ">", v)
                for u, v, _ in pattern_edges))

    if pattern_typing is not None and len(pattern_typing) > 0:
        for typing_graph, mapping in pattern_typing.items():
            query += ", {}".format(
                ", ".join(
                    "({})-[:typing*..]->({}:{})".format(
                        n, "{}_type_{}".format(n, typing_graph), typing_graph)
                    for n in pattern.nodes() if n in pattern_typing[typing_graph].keys()
                ))
    query += "\n"

    where_appeared = False
    if len(pattern_nodes) > 1:
        injectivity_clauses = []
        for i, n in enumerate(pattern_nodes):
            if i < len(pattern_nodes) - 1:
                injectivity_clauses.append(
                    "id({}) <> id({})".format(
                        pattern_nodes[i], pattern_nodes[i + 1]))
        query += "WHERE {}\n".format(
            " AND ".join(c for c in injectivity_clauses))
        where_appeared = True

    if pattern_typing is not None and len(pattern_typing) > 0:
        if where_appeared is False:
            query += "WHERE "
            where_appeared = True
        else:
            query += "AND "

        typing_strs = []
        for typing_graph, mapping in pattern_typing.items():
            typing_strs.append(" AND ".join(
                "{}.id IN toSet([{}])".format(
                    "{}_type_{}".format(n, typing_graph),
                    ", ".join("'{}'".format(t) for t in pattern_typing[typing_graph][n]))
                for n in pattern.nodes() if n in pattern_typing[typing_graph].keys()
            ))

        query += " AND ".join(typing_strs)

    if nodes is not None and len(nodes) > 0:
        if where_appeared is False:
            query += "WHERE "
            where_appeared = True
        else:
            query += "AND "
        query +=\
            " AND ".join(
                "{}.id IN toSet([{}])".format(
                    pattern_n, ", ".join("'{}'".format(n) for n in nodes))
                for pattern_n in pattern_nodes) + "\n"

    nodes_with_attrs = []
    edges_with_attrs = []
    for n, attrs in pattern.nodes(data=True):
        if len(attrs) != 0:
            for k in attrs.keys():
                for el in attrs[k]:
                    if type(el) == str:
                        nodes_with_attrs.append((n, k, "'{}'".format(el)))
                    else:
                        nodes_with_attrs.append((n, k, "{}".format(el)))

    for s, t, attrs in pattern_edges:
        if len(attrs) != 0:
            for k in attrs.keys():
                for el in attrs[k]:
                    if type(el) == str:
                        edges_with_attrs.append(
                            ("{}_to_{}".format(s, t), k, "'{}'".format(el)))
                    else:
                        edges_with_attrs.append(
                            ("{}_to_{}".format(s, t), k, "{}".format(el)))

    if len(nodes_with_attrs) != 0:
        if where_appeared is True:
            query += " AND "
        else:
            query += " WHERE "
            where_appeared = True
        query += (
            " AND ".join(["{} IN toSet([{}.{}])".format(
                v, n, k) for n, k, v in nodes_with_attrs]) + "\n"
        )
    if len(edges_with_attrs) != 0:
        if where_appeared is True:
            query += " AND "
        else:
            query += " WHERE "
            where_appeared = True
        query += (
            " AND ".join(["{} IN toSet([{}.{}])".format(
                v, e_var, k) for e_var, k, v in edges_with_attrs]) + "\n"
        )

    query += "RETURN {}".format(", ".join(["{}".format(n) for n in pattern.nodes()]))
    return query


def merging_from_list(list_var, merged_var, merged_id, merged_id_var,
                      node_label, edge_label, merge_typing=False,
                      carry_vars=None, ignore_naming=False,
                      multiple_rows=False, multiple_var=None):
    """Generate query for merging the nodes of a neo4j list.

    Parameters
    ----------
    list_var : str
        Name of the variable corresponding to the list of nodes to clone
    merged_var : str
        Name of the variable corresponding to the new merged node
    merged_id : str
        Id to use for the new node that corresponds to the merged node
    merged_id_var : str
        Name of the variable for the id of the new merged node
    node_label
        Label of the nodes to merge, default is 'node'
    edge_label
        Label of the edges to merge, default is 'edge'
    merge_typing : boolean
        If true, typing edges are merged
    carry_vars : str
        Collection of variables to carry
    multiple_rows : boolean
        Must be True when the previous matching resulted in more than one
        record (for example when there is more than one group of nodes to
        merge). When True, a map linking the original ids of the nodes to
        the id of the merged one is created so that the edges between the
        merged nodes are always preserved
    multiple_var : str
        Variable which has a different value on each row. Used for controlling
        the merging on multiple rows

    Returns
    -------
    query : str
        Generated query
    carry_vars : set
        Updated collection of variables to carry
    """
    if carry_vars is None:
        carry_vars = set([list_var])

    carry_vars.remove(list_var)
    query = "UNWIND {} as node_to_merge\n".format(list_var)

    query += (
        "// accumulate all the attrs of the nodes to be merged\n" +
        "WITH [] as new_props, node_to_merge, " + ", ".join(carry_vars) + "\n" +
        "WITH new_props + REDUCE(pairs = [], k in keys(node_to_merge) | \n" +
        "\tpairs + REDUCE(inner_pairs = [], v in node_to_merge[k] | \n" +
        "\t\tinner_pairs + {key: k, value: v})) as new_props, node_to_merge, " +
        ", ".join(carry_vars) + "\n"
    )

    query += (
        "WITH collect(node_to_merge) as {}, ".format(list_var) +
        "collect(new_props) as new_props_col, " +
        ", ".join(carry_vars) + "\n"
        "WITH REDUCE(init=[], props in new_props_col | init + props) as new_props, " +
        "{}, ".format(list_var) + ", ".join(carry_vars) + "\n"
    )
    carry_vars.add(list_var)

    query += (
        "WITH apoc.map.groupByMulti(new_props, 'key') as new_props, " +
        ", ".join(carry_vars) + "\n" +
        "WITH apoc.map.fromValues(REDUCE(pairs=[], k in keys(new_props) | \n"
        "\tpairs + [k, REDUCE(values=[], v in new_props[k] | \n"
        "\t\tvalues + CASE WHEN v.value IN values THEN [] ELSE v.value END)])) as new_props, " +
        ", ".join(carry_vars) + "\n" +
        "WITH {}[0] as {}, new_props, ".format(list_var, merged_var) +
        ", ".join(carry_vars) + "\n"
        "SET {} = new_props\n".format(merged_var)
    )
    carry_vars.add(merged_var)

    if ignore_naming is True:
        query += (
            "// set appropriate node id\n"
            "SET {}.id = toString(id({}))\n".format(merged_var, merged_var) +
            "SET {}.count = NULL\n".format(merged_var) +
            "WITH toString(id({})) as {}, ".format(merged_var, merged_id_var) +
            ", ".join(carry_vars) + "\n"
        )
    else:
        query += (
            "// search for a node with the same id as the clone id\n" +
            "OPTIONAL MATCH (same_id_node:{} {{ id : '{}'}}) \n".format(
                node_label, merged_id) +
            "WITH same_id_node,  " +
            "CASE WHEN same_id_node IS NOT NULL "
            "THEN (coalesce(same_id_node.count, 0) + 1) " +
            "ELSE 0 END AS same_id_node_new_count, " +
            ", ".join(carry_vars) + "\n" +
            "// generate new id if the same id node was found\n" +
            "// and filter edges which will be removed \n" +
            "WITH same_id_node, same_id_node_new_count, " +
            "'{}' + CASE WHEN same_id_node_new_count <> 0 ".format(merged_id) +
            "THEN toString(same_id_node_new_count) ELSE '' END as {}, ".format(
                merged_id_var) +
            ", ".join(carry_vars) + "\n"
            "// set appropriate node id\n"
            "SET {}.id = {}\n".format(merged_var, merged_id_var) +
            "SET {}.count = NULL\n".format(merged_var) +
            "WITH {}, ".format(merged_id_var) + ", ".join(carry_vars) + "\n"
        )
    carry_vars.add(merged_id_var)

    if multiple_rows:
        query += "UNWIND {} AS node_to_merge\n".format(list_var)
        carry_vars.remove(list_var)
        query += (
            "//create a map from node ids to the id of the merged_node\n" +
            "WITH {{old_node:node_to_merge.id,  new_node:{}}} as id_to_merged_id, ".format(
                merged_var) +
            "node_to_merge, " + ", ".join(carry_vars) + "\n" +
            "WITH collect(id_to_merged_id) as ids_to_merged_id, collect(node_to_merge) as {}, ".format(
                list_var) + ", ".join(carry_vars) + "\n"
        )
        carry_vars.add(list_var)
        query += (
            "WITH apoc.map.groupByMulti(ids_to_merged_id, 'old_node') as ids_to_merged_id, " +
            ", ".join(carry_vars) + "\n" +
            "WITH apoc.map.fromValues(REDUCE(pairs=[], k in keys(ids_to_merged_id) | \n"
            "\tpairs + [k, REDUCE(values=[], v in ids_to_merged_id[k] | \n"
            "\t\tvalues + CASE WHEN v.new_node IN values THEN [] ELSE v.new_node END)])) as ids_to_merged_id, " +
            ", ".join(carry_vars) + "\n"
        )
        carry_vars.difference_update([multiple_var, list_var, merged_var, merged_id_var])
        query += (
            "WITH collect({{n:{}, nodes_to_merge:{}, merged_node: {}, merged_id: {}}}) ".format(
                multiple_var, list_var, merged_var, merged_id_var) +
            "as all_n, collect(ids_to_merged_id) as ids_to_merged_id, " +
            ", ".join(carry_vars) + "\n" +
            "WITH reduce(acc={}, x in ids_to_merged_id | apoc.map.merge(acc, x))" +
            " as ids_to_merged_id, all_n, " +
            ", ".join(carry_vars) + "\n" +
            "UNWIND all_n as n_maps\n" +
            "WITH n_maps.n as {}, n_maps.nodes_to_merge as {}, ".format(
                multiple_var, list_var) +
            "n_maps.merged_node as {}, n_maps.merged_id as {}, ".format(
                merged_var, merged_id_var) +
            "ids_to_merged_id, " + ", ".join(carry_vars) + "\n"
        )
        carry_vars.update([
            multiple_var, list_var, merged_var,
            merged_id_var, 'ids_to_merged_id'])

    query += "UNWIND {} AS node_to_merge\n".format(list_var)
    carry_vars.remove(list_var)

    if multiple_rows:
        query += (
            "// accumulate all the attrs of the edges incident to the merged nodes\n"
            "WITH node_to_merge, " + ", ".join(carry_vars) + "\n"
            "OPTIONAL MATCH (node_to_merge)-[out_rel:{}]->(suc)\n".format(
                edge_label) +
            "WITH CASE WHEN suc.id IN keys(ids_to_merged_id)\n" +
            "\t\tTHEN {id: id(ids_to_merged_id[suc.id][0]), neighbor: ids_to_merged_id[suc.id][0], edge: out_rel}\n" +
            "\t\tELSE {id: id(suc), neighbor: suc, edge: out_rel} END AS suc_map, " +
            "node_to_merge, " + ", ".join(carry_vars) + "\n" +
            "WITH collect(suc_map) as suc_maps, " +
            "node_to_merge, " + ", ".join(carry_vars) + "\n" +
            "OPTIONAL MATCH (pred)-[in_rel:{}]->(node_to_merge)\n".format(
                edge_label) +
            "WITH CASE WHEN pred.id IN keys(ids_to_merged_id)\n" +
            "\t\tTHEN {id: id(ids_to_merged_id[pred.id][0]), neighbor: ids_to_merged_id[pred.id][0], edge: in_rel}\n" +
            "\t\tELSE {id: id(pred), neighbor: pred, edge: in_rel} END AS pred_map, " +
            "node_to_merge, suc_maps, " + ", ".join(carry_vars) + "\n" +
            "WITH collect(pred_map) as pred_maps, " +
            "suc_maps, node_to_merge, " + ", ".join(carry_vars) + "\n"
        )
    else:
        query += (
            "// accumulate all the attrs of the edges incident to the merged nodes\n"
            "WITH [] as suc_maps, [] as pred_maps, node_to_merge, " +
            ", ".join(carry_vars) + "\n"
            "OPTIONAL MATCH (node_to_merge)-[out_rel:{}]->(suc)\n".format(
                edge_label) +
            "WITH suc_maps + collect({id: id(suc), neighbor: suc, edge: out_rel}) as suc_maps, " +
            "pred_maps, node_to_merge, " + ", ".join(carry_vars) + "\n" +
            "OPTIONAL MATCH (pred)-[in_rel:{}]->(node_to_merge)\n".format(
                edge_label) +
            "WITH pred_maps + collect({id: id(pred), neighbor: pred, edge: in_rel}) as pred_maps, " +
            "suc_maps, node_to_merge, " + ", ".join(carry_vars) + "\n"
        )
    query += (
        "WITH collect(node_to_merge) as {}, ".format(list_var) +
        "collect(node_to_merge.id) as list_ids, "
        "collect(suc_maps) as suc_maps_col, " +
        "collect(pred_maps) as pred_maps_col, " +
        ", ".join(carry_vars) + "\n"
        "WITH REDUCE(init=[], maps in suc_maps_col | init + maps) as suc_maps, " +
        "REDUCE(init=[], maps in pred_maps_col | init + maps) as pred_maps, " +
        "list_ids, {}, ".format(list_var) + ", ".join(carry_vars) + "\n"
    )
    carry_vars.add(list_var)
    carry_vars.add('list_ids')

    query += (
        "WITH apoc.map.groupByMulti(suc_maps, 'id') as suc_props, " +
        "REDUCE(list=[], map in suc_maps | \n"
        "\tlist + CASE WHEN NOT map['neighbor'] IS NULL THEN [map['neighbor']] ELSE [] END) as suc_nodes, "
        "apoc.map.groupByMulti(pred_maps, 'id') as pred_props, " +
        "REDUCE(list=[], map in pred_maps | \n"
        "\tlist + CASE WHEN NOT map['neighbor'] IS NULL THEN [map['neighbor']] ELSE [] END) as pred_nodes, " +
        "\tREDUCE(l=[], el in suc_maps + pred_maps| \n" +
        "\t\tl + CASE WHEN el['id'] IN list_ids THEN [toString(el['id'])] ELSE [] END)" +
        " as self_loops, " +
        ", ".join(carry_vars) + "\n"
    )
    carry_vars.remove('list_ids')
    carry_vars.add("self_loops")

    query += (
        "WITH suc_nodes, pred_nodes, "
        "apoc.map.fromValues(REDUCE(edge_props=[], k in keys(suc_props) | \n"
        "\tedge_props + [k, apoc.map.groupByMulti(REDUCE(props=[], el in suc_props[k] | \n"
        "\t\tprops + REDUCE(pairs=[], kk in keys(el['edge']) | \n"
        "\t\t\tpairs + REDUCE(values=[], v in el['edge'][kk] | \n"
        "\t\t\t\tvalues + {key: kk, value: v}))), 'key')])) as suc_props, \n" +
        "\tapoc.map.fromValues(REDUCE(edge_props=[], k in keys(pred_props) | \n"
        "\tedge_props + [k, apoc.map.groupByMulti(REDUCE(props=[], el in pred_props[k] | \n"
        "\t\tprops + REDUCE(pairs=[], kk in keys(el['edge']) | \n"
        "\t\t\tpairs + REDUCE(values=[], v in el['edge'][kk] | \n"
        "\t\t\t\t values + {key: kk, value: v}))), 'key')])) as pred_props,  \n" +
        "\tREDUCE(edge_props=[], k IN filter(k IN keys(suc_props) WHERE k IN self_loops) |\n"
        "\t\tedge_props + suc_props[k]) + \n"
        "\tREDUCE(edge_props=[], k IN filter(k IN keys(pred_props) WHERE k IN self_loops) |\n"
        "\t\tedge_props + pred_props[k]) as self_loop_props, " +
        ", ".join(carry_vars) + "\n" +
        "WITH suc_nodes, suc_props, pred_nodes, pred_props, " +
        "apoc.map.groupByMulti(REDUCE(pairs=[], el in self_loop_props |\n"
        "\tpairs + REDUCE(inner_pairs=[], k in keys(el['edge']) | \n"
        "\t\tinner_pairs + REDUCE(values=[], v in el['edge'][k] |\n"
        "\t\t\tvalues + {key: k, value: v}))), 'key') as self_loop_props, " +
        ", ".join(carry_vars) + "\n"
    )
    query += (
        "FOREACH(suc IN filter(suc IN suc_nodes WHERE NOT id(suc) in self_loops) |\n"
        "\tMERGE ({})-[new_rel:{}]->(suc)\n".format(merged_var, edge_label) +
        "\tSET new_rel = apoc.map.fromValues(REDUCE(pairs=[], k in keys(suc_props[toString(id(suc))]) | \n"
        "\t\t pairs + [k, REDUCE(values=[], v in suc_props[toString(id(suc))][k] | \n"
        "\t\t\tvalues + CASE WHEN v.value IN values THEN [] ELSE v.value END)])))\n"
        "FOREACH(pred IN filter(pred IN pred_nodes WHERE NOT id(pred) in self_loops) |\n"
        "\tMERGE (pred)-[new_rel:{}]->({})\n".format(edge_label, merged_var) +
        "\tSET new_rel = apoc.map.fromValues(REDUCE(pairs=[], k in keys(pred_props[toString(id(pred))]) | \n"
        "\t\t pairs + [k, REDUCE(values=[], v in pred_props[toString(id(pred))][k] | \n"
        "\t\t\tvalues + CASE WHEN v.value IN values THEN [] ELSE v.value END)])))\n"
    )
    query += (
        "// add self loop \n"
        "FOREACH(dummy in CASE WHEN length(self_loops) > 0 THEN [NULL] ELSE [] END |\n"
        "\tMERGE ({})-[new_rel:{}]->({})\n".format(merged_var,
                                                   edge_label,
                                                   merged_var) +
        "\tSET new_rel = apoc.map.fromValues(REDUCE(pairs=[], k in keys(self_loop_props) |\n"
        "\t\tpairs + [k, REDUCE(values=[], v in self_loop_props[k] |\n"
        "\t\t\tvalues + CASE WHEN v.value IN values THEN [] ELSE v.value END)])))\n"
    )
    carry_vars.remove("self_loops")

    if merge_typing:
        query += "WITH " + ", ".join(carry_vars) + "\n"
        query += "UNWIND {} AS node_to_merge\n".format(list_var)
        carry_vars.remove(list_var)

        query += (
            "// accumulate all the attrs of the edges incident to the merged nodes\n"
            "WITH [] as suc_typings, [] as pred_typings, node_to_merge, " +
            ", ".join(carry_vars) + "\n"
        )
        query += (
            "OPTIONAL MATCH (node_to_merge)-[:typing]->(suc)\n" +
            "WITH suc_typings + collect(suc) as suc_typings, node_to_merge, " +
            "pred_typings, " + ", ".join(carry_vars) + "\n" +
            "OPTIONAL MATCH (pred)-[:typing]->(node_to_merge)\n" +
            "WITH pred_typings + collect(pred) as pred_typings, node_to_merge, " +
            "suc_typings, " + ", ".join(carry_vars) + "\n"
        )
        query += (
            "WITH collect(node_to_merge) as {}, ".format(list_var) +
            "collect(suc_typings) as suc_typings_col, " +
            "collect(pred_typings) as pred_typings_col, " +
            ", ".join(carry_vars) + "\n"
            "WITH REDUCE(init=[], sucs in suc_typings_col | init + sucs) as suc_typings, " +
            "REDUCE(init=[], preds in pred_typings_col | init + preds) as pred_typings, " +
            "{}, ".format(list_var) + ", ".join(carry_vars) + "\n"
        )
        carry_vars.add(list_var)

        query += (
            "FOREACH(suc in suc_typings |\n" +
            "\tMERGE ({})-[:typing]->(suc))\n".format(merged_var) +
            "FOREACH(pred in pred_typings |\n" +
            "\tMERGE (pred)-[:typing]->({}))\n".format(merged_var)
        )

    query += (
        "WITH " + ", ".join(carry_vars) + "\n" +
        "FOREACH(node in filter(x IN {} WHERE x <> {}) |\n".format(
            list_var, merged_var) +
        "\tDETACH DELETE node)\n"
    )

    carry_vars.remove(list_var)

    return query, carry_vars