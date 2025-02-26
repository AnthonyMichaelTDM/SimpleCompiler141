use super::scan;
use pretty_assertions::assert_str_eq;

static INPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define print(x) printf(x)

void state_0(void);
void state_1(void);
void state_2(void);
void state_3(void);

// function to get the next digit
int getnextdigit(void) 
{
    int n;
    while (0 == 0) 
    {
	print("Give me a number (-1 to quit): ");
	read(n);
	if (-1 <= n && 1 >= n) 
	{
	    break;
	}
	print("I need a number that's either 0 or 1.\n");
    }

    return n;
}

// state_0 is the initial state
void state_0(void) 
{
    int a;
    a = getnextdigit();
    if (-1 == a)
    {
	print("You gave me an even number of 0's.\n");
	print("You gave me an even number of 1's.\n");
	print("I therefore accept this input.\n");
	return;
    }
    if (0 == a)
    {
	state_2();
    }
    if (1 == a)
    {
	state_1();
    }
}

void state_1(void) 
{
    int a;
    a = getnextdigit();
    if (-1 == a)
    {
	print("You gave me an even number of 0's.\n");
	print("You gave me an odd number of 1's.\n");
	print("I therefore reject this input.\n");
	return;
    }
    if (0 == a)
    {
	state_3();
    }
    if (1 == a)
    {
	state_0();
    }
}

void state_2(void) 
{
    int a;
    a = getnextdigit();
    if (-1 == a)
    {
	print("You gave me an odd number of 0's.\n");
	print("You gave me an even number of 1's.\n");
	print("I therefore reject this input.\n");
	return;
    }
    if (0 == a)
    {
	state_0();
    }
    if (1 == a)
    {
	state_3();
    }
}

void state_3(void) 
{
    int a;
    a = getnextdigit();
    if (-1 == a)
    {
	print("You gave me an odd number of 0's.\n");
	print("You gave me an odd number of 1's.\n");
	print("I therefore reject this input.\n");
	return;
    }
    if (0 == a)
    {
	state_1();
    }
    if (1 == a)
    {
	state_2();
    }
}

int main() 
{
    state_0();
}
"#;
static OUTPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define print(x) printf(x)

void cse141state_0(void);
void cse141state_1(void);
void cse141state_2(void);
void cse141state_3(void);

// function to get the next digit
int cse141getnextdigit(void) 
{
    int cse141n;
    while (0 == 0) 
    {
	print("Give me a number (-1 to quit): ");
	read(cse141n);
	if (-1 <= cse141n && 1 >= cse141n) 
	{
	    break;
	}
	print("I need a number that's either 0 or 1.\n");
    }

    return cse141n;
}

// state_0 is the initial state
void cse141state_0(void) 
{
    int cse141a;
    cse141a = cse141getnextdigit();
    if (-1 == cse141a)
    {
	print("You gave me an even number of 0's.\n");
	print("You gave me an even number of 1's.\n");
	print("I therefore accept this input.\n");
	return;
    }
    if (0 == cse141a)
    {
	cse141state_2();
    }
    if (1 == cse141a)
    {
	cse141state_1();
    }
}

void cse141state_1(void) 
{
    int cse141a;
    cse141a = cse141getnextdigit();
    if (-1 == cse141a)
    {
	print("You gave me an even number of 0's.\n");
	print("You gave me an odd number of 1's.\n");
	print("I therefore reject this input.\n");
	return;
    }
    if (0 == cse141a)
    {
	cse141state_3();
    }
    if (1 == cse141a)
    {
	cse141state_0();
    }
}

void cse141state_2(void) 
{
    int cse141a;
    cse141a = cse141getnextdigit();
    if (-1 == cse141a)
    {
	print("You gave me an odd number of 0's.\n");
	print("You gave me an even number of 1's.\n");
	print("I therefore reject this input.\n");
	return;
    }
    if (0 == cse141a)
    {
	cse141state_0();
    }
    if (1 == cse141a)
    {
	cse141state_3();
    }
}

void cse141state_3(void) 
{
    int cse141a;
    cse141a = cse141getnextdigit();
    if (-1 == cse141a)
    {
	print("You gave me an odd number of 0's.\n");
	print("You gave me an odd number of 1's.\n");
	print("I therefore reject this input.\n");
	return;
    }
    if (0 == cse141a)
    {
	cse141state_1();
    }
    if (1 == cse141a)
    {
	cse141state_2();
    }
}

int main() 
{
    cse141state_0();
}
"#;

#[test]
fn test_scanner() {
    assert_str_eq!(OUTPUT, scan(INPUT))
}
