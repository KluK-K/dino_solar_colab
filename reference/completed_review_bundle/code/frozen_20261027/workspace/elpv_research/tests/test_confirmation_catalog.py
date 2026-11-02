from build_confirmation_catalog import candidate_family


def test_actual_dense_reference_parent_family_alias_cannot_collapse_fits():
    # Actual source metadata inherit the HPO family even after adding a normal
    # memory. Same-family fitting and ablation records must remain distinguishable.
    frozen={'name':'dinov3_dense_frozen','family':'dinov3_dense_frozen',
            'architecture':'dinov3_multidepth','reference':False,'geometry_weight':.1}
    reference={**frozen,'name':'dinov3_dense_reference','reference':True}
    no_position={**reference,'name':'dinov3_reference_no_position','geometry_weight':0.}
    def identity(spec,study='dense_development_v1'):
        return candidate_family('scdd_publication_v1','internal',1.,spec,study)[1]
    assert len({identity(frozen),identity(reference),identity(no_position)})==3
    assert identity(reference)==identity(dict(reference))  # all paired seeds
    assert identity(reference)!=identity(reference,'ucf_reference_transfer_v1')
    # Even a reused display name cannot silently alias a changed specification.
    assert identity(reference)!=identity({**reference,'geometry_weight':0.})
