"""Collection of utils for generation of propagation-related Cypher queries."""
import networkx as nx

from regraph.exceptions import InvalidHomomorphism
from regraph.utils import (keys_by_value,
                           generate_new_id,
                           attrs_intersection,
                           attrs_union)
from regraph.primitives import (add_nodes_from,
                                add_edges_from,
                                get_edge,
                                get_node,
                                exists_edge)
from regraph.category_utils import (pullback,
                                    pushout,
                                    compose)
from regraph.rules import Rule

from . import generic

def get_typing(domain, codomain, typing_label, attrs=None):
    query = (
        "MATCH (n:{})-[:{}*1..]->(m:{})\n".format(
            domain, typing_label, codomain) +
        "RETURN n.id as node, m.id as type"
    )
    return query


def get_relation(domain, codomain, typing_label, attrs=None):
    query = (
        "MATCH (n:{})-[:{}]-(m:{})\n".format(
            domain, typing_label, codomain) +
        "RETURN n.id as node, m.id as type"
    )
    return query


def set_intergraph_edge(domain, codomain, domain_node,
                        codomain_node, typing_label,
                        attrs=None):
    query = (
        "MATCH (n:{} {{ id: '{}' }}), (m:{} {{ id: '{}' }})\n".format(
            domain, domain_node, codomain, codomain_node) +
        "MERGE (n)-[:{}  {{ {} }}]->(m)".format(typing_label, generic.generate_attributes(attrs))
    )
    return query


def check_homomorphism(tx, domain, codomain, total=True):
    """Check if the homomorphism is valid.

    Parameters
    ----------
    tx
        Variable of a cypher transaction
    domain : str
        Label of the graph at the domain of the homomorphism
    codmain : str
        Label of the graph at the codomain of the homomorphism

    Raises
    ------
    InvalidHomomorphism
        This error is raised in the following cases:

            * a node at the domain does not have exactly 1 image
            in the codoamin
            * an edge at the domain does not have an image in
            the codomain
            * a property does not match between a node and its image
            * a property does not match between an edge and its image
    """
    # Check if all the nodes of the domain have exactly 1 image
    query1 = (
        "MATCH (n:{})\n".format(domain) +
        "OPTIONAL MATCH (n)-[:typing]->(m:{})\n".format(codomain) +
        "WITH n, collect(m) as images\n" +
        "WHERE size(images) <> 1\n" +
        "RETURN n.id as ids, size(images) as nb_of_img\n"
    )

    result = tx.run(query1)
    nodes = []

    for record in result:
        nodes.append((record['ids'], record['nb_of_img']))
    if len(nodes) != 0:
        raise InvalidHomomorphism(
            "Wrong number of images!\n" +
            "\n".join(
                ["The node '{}' of the graph {} have {} image(s) in the graph {}.".format(
                    n, domain, str(nb), codomain) for n, nb in nodes]))

    # Check if all the edges of the domain have an image
    query2 = (
        "MATCH (n:{})-[:edge]->(m:{})\n".format(
            domain, domain) +
        "MATCH (n)-[:typing]->(x:{}), (y:{})<-[:typing]-(m)\n".format(
            codomain, codomain) +
        "OPTIONAL MATCH (x)-[r:edge]->(y)\n" +
        "WITH x.id as x_id, y.id as y_id, r\n" +
        "WHERE r IS NULL\n" +
        "WITH x_id, y_id, collect(r) as rs\n" +
        "RETURN x_id, y_id\n"
    )

    result = tx.run(query2)
    xy_ids = []
    for record in result:
        xy_ids.append((record['x_id'], record['y_id']))
    if len(xy_ids) != 0:
        raise InvalidHomomorphism(
            "Edges are not preserved in the homomorphism from '{}' to '{}': ".format(
                domain, codomain) +
            "Was expecting edges {}".format(
                ", ".join(
                    "'{}'->'{}'".format(x, y) for x, y in xy_ids))
        )

    # "CASE WHEN size(apoc.text.regexGroups(m_props, 'IntegerSet\\[(\\d+|minf)-(\\d+|inf)\\]') AS value"

    # Check if all the attributes of a node of the domain are in its image
    query3 = (
        "MATCH (n:{})-[:typing]->(m:{})\n".format(
            domain, codomain) +
        "WITH properties(n) as n_props, properties(m) as m_props, " +
        "n.id as n_id, m.id as m_id\n" +
        "WITH REDUCE(invalid = 0, k in filter(k in keys(n_props) WHERE k <> 'id' AND k <> 'count') |\n" +
        "\tinvalid + CASE\n" +
        "\t\tWHEN NOT k IN keys(m_props) THEN 1\n" +
        "\t\tELSE REDUCE(invalid_values = 0, v in n_props[k] |\n" +
        "\t\t\tinvalid_values + CASE m_props[k]\n" +
        "\t\t\t\tWHEN ['IntegerSet'] THEN CASE WHEN toInt(v) IS NULL THEN 1 ELSE 0 END\n" +
        "\t\t\t\tWHEN ['StringSet'] THEN CASE WHEN toString(v) <> v THEN 1 ELSE 0 END\n" +
        "\t\t\t\tWHEN ['BooleanSet'] THEN CASE WHEN v=true OR v=false THEN 0 ELSE 1 END\n" +
        "\t\t\t\tELSE CASE WHEN NOT v IN m_props[k] THEN 1 ELSE 0 END END)\n" +
        "\t\tEND) AS invalid, n_id, m_id\n" +
        "WHERE invalid <> 0\n" +
        "RETURN n_id, m_id, invalid\n"
    )

    result = tx.run(query3)
    invalid_typings = []
    for record in result:
        invalid_typings.append((record['n_id'], record['m_id']))
    if len(invalid_typings) != 0:
        raise InvalidHomomorphism(
            "Node attributes are not preserved in the homomorphism from '{}' to '{}': ".format(
                domain, codomain) +
            "\n".join(["Attributes of nodes source: '{}' ".format(n) +
                       "and target: '{}' do not match!".format(m)
                       for n, m in invalid_typings]))

    # Check if all the attributes of an edge of the domain are in its image
    query4 = (
        "MATCH (n:{})-[rel_orig:edge]->(m:{})\n".format(
            domain, domain) +
        "MATCH (n)-[:typing]->(x:{}), (y:{})<-[:typing]-(m)\n".format(
            codomain, codomain) +
        "MATCH (x)-[rel_img:edge]->(y)\n" +
        "WITH n.id as n_id, m.id as m_id, x.id as x_id, y.id as y_id, " +
        "properties(rel_orig) as rel_orig_props, " +
        "properties(rel_img) as rel_img_props\n" +
        "WITH REDUCE(invalid = 0, k in keys(rel_orig_props) |\n" +
        "\tinvalid + CASE\n" +
        "\t\tWHEN NOT k IN keys(rel_img_props) THEN 1\n" +
        "\t\tELSE REDUCE(invalid_values = 0, v in rel_orig_props[k] |\n" +
        "\t\t\tinvalid_values + CASE rel_img_props[k]\n" +
        "\t\t\t\tWHEN ['IntegerSet'] THEN CASE WHEN toInt(v) IS NULL THEN 1 ELSE 0 END\n" +
        "\t\t\t\tWHEN ['StringSet'] THEN CASE WHEN toString(v) <> v THEN 1 ELSE 0 END\n" +
        "\t\t\t\tWHEN ['BooleanSet'] THEN CASE WHEN v=true OR v=false THEN 0 ELSE 1 END\n" +
        "\t\t\t\tELSE CASE WHEN NOT v IN rel_img_props[k] THEN 1 ELSE 0 END END)\n" +
        "\t\tEND) AS invalid, n_id, m_id, x_id, y_id\n" +
        "WHERE invalid <> 0\n" +
        "RETURN n_id, m_id, x_id, y_id, invalid\n"
    )
    result = tx.run(query4)
    invalid_edges = []
    for record in result:
        invalid_edges.append((record['n_id'], record['m_id'],
                              record['x_id'], record['y_id']))
    if len(invalid_edges) != 0:
        raise InvalidHomomorphism(
            "Edge attributes are not preserved!\n" +
            "\n".join(["Attributes of edges '{}'->'{}' ".format(n, m) +
                       "and '{}'->'{}' do not match!".format(x, y)
                       for n, m, x, y in invalid_edges])
        )

    return True


def check_consistency(tx, source, target):
    """Check if the adding of a homomorphism is consistent."""
    query = (
        "// match all typing pairs between '{}' and '{}'\n".format(
            source, target) +
        "MATCH (s:{})-[:typing]->(t:{})\n".format(
            source, target) +
        "WITH s, t\n"
    )
    query += (
        "// match all the predecessors of 's' and successors of 't'\n"
        "MATCH (pred)-[:typing*0..]->(s), (t)-[:typing*0..]->(suc) \n"
        "WHERE NOT pred = s AND NOT suc = t\n" +
        "WITH s, t, collect(DISTINCT pred) as pred_list, " +
        "collect(DISTINCT suc) as suc_list\n"
    )
    query += (
        "// select all the pairs 'pred' 'suc' with a path between\n"
        "UNWIND pred_list as pred\n" +
        "UNWIND suc_list as suc\n" +
        "OPTIONAL MATCH (pred)-[r:typing*]->(suc)\n" +
        "WHERE NONE(rel in r WHERE rel.tmp = 'True')\n"
        "WITH s, t, r, labels(pred)[1] as pred_label, labels(suc)[1] as suc_label\n" +
        "WHERE r IS NOT NULL\n" +
        "WITH DISTINCT s, t, pred_label, suc_label\n"
    )
    query += (
        "// return the pairs 's' 't' where there should be a typing edge\n"
        "OPTIONAL MATCH (s)-[new_typing:typing]->(t)\n" +
        "WHERE new_typing.tmp IS NOT NULL\n" +
        "WITH pred_label, suc_label, s.id as s_id, t.id as t_id, new_typing\n" +
        "WHERE new_typing IS NULL\n" +
        "RETURN pred_label, suc_label, s_id, t_id\n"
    )
    result = tx.run(query)

    missing_typing = []
    for record in result:
        missing_typing.append((record['pred_label'], record['suc_label']))
    if len(missing_typing) != 0:
        raise InvalidHomomorphism(
            "Homomorphism does not commute with existing paths:\n" +
            ",\n".join(["\t- from {} to {}".format(
                s, t) for s, t in missing_typing]) + "."
        )

    return True


def get_rule_liftings(tx, graph_id, rule, instance, p_typing=None):
    """Execute the query finding rule liftings."""
    if p_typing is None:
        p_typing = {}

    liftings = {}
    if len(rule.lhs.nodes()) > 0:
        lhs_vars = {
            n: n for n in rule.lhs.nodes()}
        match_instance_vars = {lhs_vars[k]: v for k, v in instance.items()}

        # Match nodes
        query = "// Match nodes the instance of the rewritten graph \n"
        query += "MATCH {}".format(
            ", ".join([
                "({}:{} {{id: '{}'}})".format(k, graph_id, v)
                for k, v in match_instance_vars.items()
            ])
        )
        query += "\n\n"

        carry_vars = list(lhs_vars.values())
        for k, v in lhs_vars.items():
            query += (
                "OPTIONAL MATCH (n)-[:typing*1..]->({})\n".format(v) +
                "WITH {} \n".format(
                    ", ".join(carry_vars + [
                        "collect({{type:'node', origin: {}.id, id: n.id, graph:labels(n)[0], attrs: properties(n)}}) as {}_dict\n".format(
                            v, v)])
                )
            )
            carry_vars.append("{}_dict".format(v))
        # Match edges
        for (u, v) in rule.lhs.edges():
            edge_var = "{}_{}".format(lhs_vars[u], lhs_vars[v])
            query += "OPTIONAL MATCH ({}_instance)-[{}:edge]->({}_instance)\n".format(
                lhs_vars[u],
                edge_var,
                lhs_vars[v])
            query += "WHERE ({})-[:typing*1..]->({}) AND ({})-[:typing*1..]->({})\n".format(
                "{}_instance".format(lhs_vars[u]), lhs_vars[u],
                "{}_instance".format(lhs_vars[v]), lhs_vars[v])
            query += (
                "WITH {} \n".format(
                    ", ".join(carry_vars + [
                        "collect({{type: 'edge', source: {}.id, target: {}.id, attrs: properties({}), graph:labels({})[0]}}) as {}\n".format(
                            "{}_instance".format(lhs_vars[u]),
                            "{}_instance".format(lhs_vars[v]),
                            edge_var,
                            "{}_instance".format(lhs_vars[u]),
                            edge_var)
                    ])
                )
            )
            carry_vars.append(edge_var)
        query += "RETURN {}".format(
            ", ".join(
                ["{}_dict as {}".format(v, v) for v in lhs_vars.values()] +
                ["{}_{}".format(lhs_vars[u], lhs_vars[v]) for u, v in rule.lhs.edges()]))

        result = tx.run(query)
        record = result.single()
        l_g_ls = {}
        lhs_nodes = {}
        lhs_edges = {}
        for k, v in record.items():
            if len(v) > 0:
                if v[0]["type"] == "node":
                    for el in v:
                        if el["graph"] not in lhs_nodes:
                            lhs_nodes[el["graph"]] = []
                            l_g_ls[el["graph"]] = {}
                        l_g_ls[el["graph"]][el["id"]] = keys_by_value(
                            instance, el["origin"])[0]
                        # compute attr intersection
                        attrs = attrs_intersection(
                            generic.convert_props_to_attrs(el["attrs"]),
                            get_node(rule.lhs, l_g_ls[el["graph"]][el["id"]]))
                        lhs_nodes[el["graph"]].append((el["id"], attrs))

                else:
                    for el in v:
                        if el["graph"] not in lhs_edges:
                            lhs_edges[el["graph"]] = []
                        # compute attr intersection
                        attrs = attrs_intersection(
                            generic.convert_props_to_attrs(el["attrs"]),
                            get_edge(
                                rule.lhs,
                                l_g_ls[el["graph"]][el["source"]],
                                l_g_ls[el["graph"]][el["target"]]))
                        lhs_edges[el["graph"]].append(
                            (el["source"], el["target"], attrs)
                        )

        for graph, nodes in lhs_nodes.items():

            lhs = nx.DiGraph()
            add_nodes_from(lhs, nodes)
            if graph in lhs_edges:
                add_edges_from(
                    lhs, lhs_edges[graph])

            p, p_lhs, p_g_p = pullback(
                lhs, rule.p, rule.lhs, l_g_ls[graph], rule.p_lhs)

            l_g_g = {n[0]: n[0] for n in nodes}

            # Remove controlled things from P_G
            if graph in p_typing.keys():
                l_g_factorization = {
                    keys_by_value(l_g_g, k)[0]: v
                    for k, v in p_typing[graph].items()
                }
                p_g_nodes_to_remove = set()
                for n in p.nodes():
                    l_g_node = p_lhs[n]
                    # If corresponding L_G node is specified in
                    # the controlling relation, remove all
                    # the instances of P nodes not mentioned
                    # in this relations
                    if l_g_node in l_g_factorization.keys():
                        p_nodes = l_g_factorization[l_g_node]
                        if p_g_p[n] not in p_nodes:
                            del p_g_p[n]
                            del p_lhs[n]
                            p_g_nodes_to_remove.add(n)

                for n in p_g_nodes_to_remove:
                    p.remove_node(n)

            liftings[graph] = {
                "rule": Rule(p=p, lhs=lhs, p_lhs=p_lhs),
                "instance": l_g_g,
                "l_g_l": l_g_ls[graph],
                "p_g_p": p_g_p
            }
    else:
        query = generic.ancestors_query(graph_id, "graph", "homomorphism")
        result = tx.run(query)
        ancestors = [record["ancestor"] for record in result]
        for a in ancestors:
            liftings[a] = {
                "rule": Rule.identity_rule(),
                "instance": {},
                "l_g_l": {},
                "p_g_p": {}
            }

    return liftings


def get_rule_projections(tx, hierarchy, graph_id, rule, instance, rhs_typing=None):
    """Execute the query finding rule liftings."""
    if rhs_typing is None:
        rhs_typing = {}

    projections = {}

    if rule.is_relaxing():
        if len(rule.lhs.nodes()) > 0:
            lhs_instance = {
                n: instance[n] for n in rule.lhs.nodes()
            }
            lhs_vars = {
                n: n for n in rule.lhs.nodes()}
            match_instance_vars = {
                v: lhs_instance[k] for k, v in lhs_vars.items()
            }

            # Match nodes
            query = "// Match nodes the instance of the rewritten graph \n"
            query += "MATCH {}".format(
                ", ".join([
                    "({}:{} {{id: '{}'}})".format(k, graph_id, v)
                    for k, v in match_instance_vars.items()
                ])
            )
            query += "\n\n"

            carry_vars = list(lhs_vars.values())
            for k, v in lhs_vars.items():
                query += (
                    "OPTIONAL MATCH (n)<-[:typing*1..]-({})\n".format(v) +
                    "WITH {} \n".format(
                        ", ".join(
                            carry_vars +
                            ["collect(DISTINCT {{type:'node', origin: {}.id, id: n.id, graph:labels(n)[0], attrs: properties(n)}}) as {}_dict\n".format(
                                v, v)])
                    )
                )
                carry_vars.append("{}_dict".format(v))

            # Match edges
            for (u, v) in rule.p.edges():
                edge_var = "{}_{}".format(lhs_vars[u], lhs_vars[v])
                query += "OPTIONAL MATCH ({}_instance)-[{}:edge]->({}_instance)\n".format(
                    lhs_vars[u],
                    edge_var,
                    lhs_vars[v])
                query += "WHERE ({})<-[:typing*1..]-({}) AND ({})<-[:typing*1..]-({})\n".format(
                    "{}_instance".format(lhs_vars[u]), lhs_vars[u],
                    "{}_instance".format(lhs_vars[v]), lhs_vars[v])
                query += (
                    "WITH {} \n".format(
                        ", ".join(carry_vars + [
                            "collect({{type: 'edge', source: {}.id, target: {}.id, graph:labels({})[0], attrs: properties({})}}) as {}\n".format(
                                "{}_instance".format(lhs_vars[u]),
                                "{}_instance".format(lhs_vars[v]),
                                "{}_instance".format(lhs_vars[u]),
                                edge_var,
                                edge_var)
                        ])
                    )
                )
                carry_vars.append(edge_var)
            query += "RETURN {}".format(
                ", ".join(
                    ["{}_dict as {}".format(v, v) for v in lhs_vars.values()] +
                    ["{}_{}".format(lhs_vars[u], lhs_vars[v]) for u, v in rule.p.edges()]))

            result = tx.run(query)
            record = result.single()

            l_l_ts = {}
            l_nodes = {}
            l_edges = {}
            for k, v in record.items():
                if len(v) > 0:
                    if v[0]["type"] == "node":
                        for el in v:
                            l_node = keys_by_value(instance, el["origin"])[0]
                            if el["graph"] not in l_nodes:
                                l_nodes[el["graph"]] = {}
                                l_l_ts[el["graph"]] = {}
                            if el["id"] not in l_nodes[el["graph"]]:
                                l_nodes[el["graph"]][el["id"]] = {}
                            l_nodes[el["graph"]][el["id"]] = attrs_union(
                                l_nodes[el["graph"]][el["id"]],
                                attrs_intersection(
                                    generic.convert_props_to_attrs(el["attrs"]),
                                    get_node(rule.lhs, l_node)))
                            l_l_ts[el["graph"]][l_node] = el["id"]
                    else:
                        for el in v:
                            l_sources = keys_by_value(l_l_ts[el["graph"]], el["source"])
                            l_targets = keys_by_value(l_l_ts[el["graph"]], el["target"])

                            for l_source in l_sources:
                                for l_target in l_targets:
                                    if exists_edge(rule.l, l_source, l_target):
                                        if el["graph"] not in l_edges:
                                            l_edges[el["graph"]] = {}
                                        if (el["source"], el["target"]) not in l_edges[el["graph"]]:
                                            l_edges[el["graph"]][(el["source"], el["target"])] = {}
                                        l_edges[el["graph"]][(el["source"], el["target"])] =\
                                            attrs_union(
                                                l_edges[el["graph"]][(el["source"], el["target"])],
                                                attrs_intersection(
                                                    generic.convert_props_to_attrs(el["attrs"]),
                                                    get_edge(rule.lhs, l_source, l_target)))

        for graph, typing in hierarchy.get_descendants(graph_id).items():
            if graph in l_nodes:
                nodes = l_nodes[graph]
            else:
                nodes = {}
            if graph in l_edges:
                edges = l_edges[graph]
            else:
                edges = {}

            l = nx.DiGraph()
            add_nodes_from(l, [(k, v) for k, v in nodes.items()])
            if graph in l_edges:
                add_edges_from(
                    l,
                    [(s, t, v) for (s, t), v in edges.items()])

            rhs, p_rhs, r_r_t = pushout(
                rule.p, l, rule.rhs, compose(rule.p_lhs, l_l_ts[graph]), rule.p_rhs)

            l_t_t = {n: n for n in nodes}

            # Modify P_T and R_T according to the controlling
            # relation rhs_typing
            if graph in rhs_typing.keys():
                r_t_factorization = {
                    r_r_t[k]: v
                    for k, v in rhs_typing[graph].items()
                }
                added_t_nodes = set()
                for n in rhs.nodes():
                    if n in r_t_factorization.keys():
                        # If corresponding R_T node is specified in
                        # the controlling relation add nodes of T
                        # that type it to P
                        t_nodes = r_t_factorization[n]
                        for t_node in t_nodes:
                            if t_node not in l_t_t.values() and\
                               t_node not in added_t_nodes:
                                new_p_node = generate_new_id(
                                    l.nodes(), t_node)
                                l.add_node(new_p_node)
                                added_t_nodes.add(t_node)
                                p_rhs[new_p_node] = n
                                l_t_t[new_p_node] = t_node
                            else:
                                p_rhs[keys_by_value(l_t_t, t_node)[0]] = n

            projections[graph] = {
                "rule": Rule(p=l, rhs=rhs, p_rhs=p_rhs),
                "instance": l_t_t,
                "l_l_t": l_l_ts[graph],
                "p_p_t": {k: l_l_ts[graph][v] for k, v in rule.p_lhs.items()},
                "r_r_t": r_r_t
            }

    return projections
