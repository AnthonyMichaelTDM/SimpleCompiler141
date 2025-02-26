use super::scan;
use pretty_assertions::assert_str_eq;

static INPUT: &'static str = r#"
%m:%~->tcsh

!#/bin/csh

>==

====


"#;
static OUTPUT: &'static str = r#"
The input program contains errors for scanning."#;

#[test]
fn test_scanner() {
    assert_str_eq!(OUTPUT, scan(INPUT))
}
