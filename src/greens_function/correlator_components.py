def get_two_point_operator_list(spin):
    """return the ordering of operators for all components of
    a two point correlator
    $<T_c\{c_{\sigma}(t_1)c^{\dager'}_{\sigma}(t_2)>$

    Parameters
    ----------
    spin : Tuple[str,str]
        Contains the desired spins, e.g. ('up','up')

    Returns
    -------
    out: dict
        contains the ordering of operators in the correlator for all
        components
    """
    correlator_components = {}

    correlator_components[(0, 0)] = [(('c', spin[0]),
                                      ('cdag', spin[1])),
                                     (('cdag', spin[1]),
                                         ('c', spin[0]))]

    correlator_components[(1, 0)] = [(('cdag', spin[1]),
                                      ('cdag_tilde', spin[0])),
                                     (('c', spin[0]),
                                         ('cdag', spin[1]))]

    correlator_components[(0, 1)] = [(('cdag', spin[1]),
                                      ('c', spin[0])),
                                     (('c', spin[0]),
                                         ('c_tilde', spin[1]))]

    correlator_components[(1, 1)] = [(('c', spin[0]),
                                      ('c_tilde', spin[1])),
                                     (('cdag', spin[1]),
                                         ('cdag_tilde', spin[0]))
                                     ]
    return correlator_components


def get_three_point_operator_list(spin):
    """return the ordering of operators for all components of
    a three point correlator
    $<T_c\{c_{\sigma}(t_1)c^{\dager'}_{\sigma}(t_2)\rho^{\chi}(t_3)\}>$

    Parameters
    ----------
    spin : Tuple[str,str,str]
        Contains the desired spins and channels, e.g. ('up','up','ch')

    Returns
    -------
    out: dict
        contains the ordering of operators in the correlator for all
        components
    """
    correlator_components = {}

    correlator_components[(0, 0, 0)] = [(('c', spin[0]),
                                         ('cdag', spin[1]),
                                         ('n_channel', spin[2])),

                                        (('c', spin[0]),
                                            ('n_channel', spin[2]),
                                            ('cdag', spin[1])),

                                        (('cdag', spin[1]),
                                            ('c', spin[0]),
                                            ('n_channel', spin[2])),

                                        (('cdag', spin[1]),
                                            ('n_channel', spin[2]),
                                            ('c', spin[0])),

                                        (('n_channel', spin[2]),
                                            ('c', spin[0]),
                                            ('cdag', spin[1])),

                                        (('n_channel', spin[2]),
                                            ('cdag', spin[1]),
                                            ('c', spin[0]))]

    correlator_components[(1, 0, 0)] = [(('c', spin[0]),
                                         ('cdag', spin[1]),
                                         ('n_channel', spin[2])),

                                        (('cdag', spin[1]),
                                            ('n_channel', spin[2]),
                                            ('cdag_tilde', spin[0])),

                                        (('c', spin[0]),
                                            ('n_channel', spin[2]),
                                            ('cdag', spin[1])),

                                        (('n_channel', spin[2]),
                                            ('cdag', spin[1]),
                                            ('cdag_tilde', spin[0]))
                                        ]

    correlator_components[(0, 1, 0)] = [(('cdag', spin[1]),
                                         ('c', spin[0]),
                                         ('n_channel', spin[2])),

                                        (('c', spin[0]),
                                            ('n_channel', spin[2]),
                                            ('c_tilde', spin[1])),

                                        (('cdag', spin[1]),
                                            ('n_channel', spin[2]),
                                            ('c', spin[0])),

                                        (('n_channel', spin[2]),
                                            ('c', spin[0]),
                                            ('c_tilde', spin[1]))]

    correlator_components[(0, 0, 1)] = [(('n_channel', spin[2]),
                                         ('c', spin[0]),
                                         ('cdag', spin[1])),

                                        (('c', spin[0]),
                                            ('cdag', spin[1]),
                                            ('n_channel_tilde', spin[2])),

                                        (('n_channel', spin[2]),
                                            ('cdag', spin[1]),
                                            ('c', spin[0])),

                                        (('cdag', spin[1]),
                                            ('c', spin[0]),
                                            ('n_channel_tilde', spin[2]))
                                        ]

    correlator_components[(1, 1, 0)] = [(('c', spin[0]),
                                         ('cdag', spin[1]),
                                         ('n_channel', spin[2])),

                                        (('cdag', spin[1]),
                                            ('n_channel', spin[2]),
                                            ('cdag_tilde', spin[0])),

                                        (('cdag', spin[1]),
                                            ('n_channel', spin[2]),
                                            ('cdag_tilde', spin[0])),

                                        (('cdag', spin[1]),
                                            ('c', spin[0]),
                                            ('n_channel', spin[2])),

                                        (('c', spin[0]),
                                            ('n_channel', spin[2]),
                                            ('c_tilde', spin[1])),

                                        (('c', spin[0]),
                                            ('n_channel', spin[2]),
                                            ('c_tilde', spin[1]))]

    correlator_components[(1, 0, 1)] = [(('c', spin[0]),
                                         ('n_channel', spin[2]),
                                         ('cdag', spin[1])),

                                        (('n_channel', spin[2]),
                                            ('cdag', spin[1]),
                                            ('cdag_tilde', spin[0])),

                                        (('n_channel', spin[2]),
                                            ('cdag', spin[1]),
                                            ('cdag_tilde', spin[0])),

                                        (('n_channel', spin[2]),
                                            ('c', spin[0]),
                                            ('cdag', spin[1])),

                                        (('c', spin[0]),
                                            ('cdag', spin[1]),
                                            ('n_channel_tilde', spin[2])),

                                        (('c', spin[0]),
                                            ('cdag', spin[1]),
                                            ('n_channel_tilde', spin[2]))]

    correlator_components[(0, 1, 1)] = [(('cdag', spin[1]),
                                         ('n_channel', spin[2]),
                                         ('c', spin[0])),

                                        (('n_channel', spin[2]),
                                            ('c', spin[0]),
                                            ('c_tilde', spin[1])),

                                        (('n_channel', spin[2]),
                                            ('c', spin[0]),
                                            ('c_tilde', spin[1])),

                                        (('n_channel', spin[2]),
                                            ('cdag', spin[1]),
                                            ('c', spin[0])),

                                        (('cdag', spin[1]),
                                            ('c', spin[0]),
                                            ('n_channel_tilde', spin[2])),

                                        (('cdag', spin[1]),
                                            ('c', spin[0]),
                                            ('n_channel_tilde', spin[2]))
                                        ]

    correlator_components[(1, 1, 1)] = [(('cdag', spin[1]),
                                         ('n_channel', spin[2]),
                                         ('cdag_tilde', spin[0])),

                                        (('n_channel', spin[2]),
                                            ('cdag', spin[1]),
                                            ('cdag_tilde', spin[0])),

                                        (('c', spin[0]),
                                            ('n_channel', spin[2]),
                                            ('c_tilde', spin[1])),

                                        (('n_channel', spin[2]),
                                            ('c', spin[0]),
                                            ('c_tilde', spin[1])),

                                        (('c', spin[0]),
                                            ('cdag', spin[1]),
                                            ('n_channel_tilde', spin[2])),

                                        (('cdag', spin[1]),
                                            ('c', spin[0]),
                                            ('n_channel_tilde', spin[2]))
                                        ]
    return correlator_components
