use multiplierless_rs::FilterDesignConstruct;

#[test]
fn test_default_n32() {
    let fdc = FilterDesignConstruct::new_default(32);
    assert_eq!(fdc.n, 32);
    assert!(fdc.lpsq > 0.0);
    assert!(fdc.upsq > 0.0);
    assert!(fdc.spsq > 0.0);
    assert!(fdc.ap.rows() > 0);
    assert_eq!(fdc.ap.cols(), 32);
    assert!(fdc.as_.rows() > 0);
    assert!(fdc.anr.rows() > 0);
}

#[test]
fn test_custom_parameters() {
    let fdc = FilterDesignConstruct::new(32, 0.15, 0.25, 0.1, 0.1, 10);
    assert_eq!(fdc.n, 32);
    assert!(fdc.lpsq > 0.0);
    assert!(fdc.upsq > fdc.lpsq);
    assert!(fdc.spsq > 0.0);
    assert!(fdc.ap.rows() > 0);
    assert!(fdc.as_.rows() > 0);
    assert!(fdc.anr.rows() > 0);
}

#[test]
fn test_smaller_order() {
    let fdc = FilterDesignConstruct::new_default(8);
    assert_eq!(fdc.n, 8);
    assert_eq!(fdc.ap.cols(), 8);
}

#[test]
fn test_different_passband_stopband() {
    let fdc = FilterDesignConstruct::new(16, 0.08, 0.30, 0.05, 0.05, 12);
    assert_eq!(fdc.n, 16);
    assert!(fdc.lpsq > 0.0);
    assert!(fdc.upsq > 0.0);
    assert!(fdc.spsq > 0.0);

    let fdc2 = FilterDesignConstruct::new(16, 0.30, 0.40, 0.2, 0.01, 12);
    assert_eq!(fdc2.n, 16);
    assert!(fdc2.lpsq > 0.0);
    assert!(fdc2.upsq > 0.0);
    assert!(fdc2.spsq > 0.0);
}

#[test]
fn test_tight_ripple_specs() {
    let fdc = FilterDesignConstruct::new(64, 0.2, 0.24, 0.01, 0.01, 20);
    assert_eq!(fdc.n, 64);
    assert!(fdc.lpsq > 0.0);
    assert!(fdc.upsq > fdc.lpsq);
    assert!(fdc.ap.rows() > 0);
    assert!(fdc.as_.rows() > 0);
}
