use super::scan;
use pretty_assertions::assert_str_eq;

static INPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define print(x) printf(x)

// recursedigit(int n) function
void recursedigit(int n) {
    int on;
    if (0 == n) {
	return;
    }
    on = 0;
    if (0 != (n-((n/2)*2))) {
        on = 1;
    }
    recursedigit(n/2);
    if (0 == on) {
	print("0");
    }
    if (1 == on) {
	print("1");
    }
}

// the entry point
int main() {
    int a;
    a = 0;
    while (0 >= a) {
	print("Give me a number: ");
	read(a);
	
	if (0 >= a) {
	    print("I need a positive integer.\n");
	}
    }
    print("The binary representation of: ");
    write(a);
    print("is: ");
    recursedigit(a);
    print("\n\n");
}


"#;
static OUTPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define print(x) printf(x)

// recursedigit(int n) function
void cse141recursedigit(int cse141n) {
    int cse141on;
    if (0 == cse141n) {
	return;
    }
    cse141on = 0;
    if (0 != (cse141n-((cse141n/2)*2))) {
        cse141on = 1;
    }
    cse141recursedigit(cse141n/2);
    if (0 == cse141on) {
	print("0");
    }
    if (1 == cse141on) {
	print("1");
    }
}

// the entry point
int main() {
    int cse141a;
    cse141a = 0;
    while (0 >= cse141a) {
	print("Give me a number: ");
	read(cse141a);
	
	if (0 >= cse141a) {
	    print("I need a positive integer.\n");
	}
    }
    print("The binary representation of: ");
    write(cse141a);
    print("is: ");
    cse141recursedigit(cse141a);
    print("\n\n");
}


"#;

#[test]
fn test_scanner() {
    assert_str_eq!(OUTPUT, scan(INPUT))
}
