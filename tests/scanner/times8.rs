use pretty_assertions::assert_str_eq;
use super::scan;

static INPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)

int add(int a, int b) {
  return a+b;
}

int times_eight(int a) {
  return add(add(add(a,a),add(a,a)), add(add(a,a),add(a,a)));
}

int main() {
    int a, b;
    read(a);
    write(times_eight(a));
}
"#;
static OUTPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)

int cse141add(int cse141a, int cse141b) {
  return cse141a+cse141b;
}

int cse141times_eight(int cse141a) {
  return cse141add(cse141add(cse141add(cse141a,cse141a),cse141add(cse141a,cse141a)), cse141add(cse141add(cse141a,cse141a),cse141add(cse141a,cse141a)));
}

int main() {
    int cse141a, cse141b;
    read(cse141a);
    write(cse141times_eight(cse141a));
}
"#;

#[test]
fn test_scanner() {
    assert_str_eq!(OUTPUT, scan(INPUT))
}
