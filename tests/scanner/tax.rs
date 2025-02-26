use pretty_assertions::assert_str_eq;
use super::scan;

static INPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define print(x) printf(x)

int getinput(void)
{
    int a;
    a = -1;
    while (0 > a)
    {
	read(a);
	if (0 > a)
	{
	    print("I need a non-negative number: ");
	}
    }

    return a;
}

int main() 
{
    int line1, line2, line3, line4, line5, line6, line7, line8, line9, 
	line10, line11, line12, dependant, single, a, b, c, d, e, f, g, 
	eic, spousedependant;

    print("Welcome to the United States 1040 federal income tax program.\n");
    print("(Note: this isn't the real 1040 form. If you try to submit your\n");
    print("taxes this way, you'll get what you deserve!\n\n");

    print("Answer the following questions to determine what you owe.\n\n");

    print("Total wages, salary, and tips? ");
    line1 = getinput();
    print("Taxable interest (such as from bank accounts)? ");
    line2 = getinput();
    print("Unemployment compensation, qualified state tuition, and Alaska\n");
    print("Permanent Fund dividends? ");
    line3 = getinput();
    line4 = line1+line2+line3;
    print("Your adjusted gross income is: ");
    write(line4);

    print("Enter <1> if your parents or someone else can claim you on their");
    print(" return. \nEnter <0> otherwise: ");
    dependant = getinput();
    if (0 != dependant)
    {
	a = line1 + 250;
	b = 700;
	c = b;
	if (c < a)
	{
	    c = a;
	}
	print("Enter <1> if you are single, <0> if you are married: ");
	single = getinput();
	if (0 != single)
	{
	    d = 7350;
	}
	if (0 == single)
	{
	    d = 4400;
	}
	e = c;
	if (e > d)
	{
	    e = d;
	}
	f = 0;
	if (single == 0)
	{
	    print("Enter <1> if your spouse can be claimed as a dependant, ");
	    print("enter <0> if not: ");
	    spousedependant = getinput();
	    if (0 == spousedependant)
	    {
		f = 2800;
	    }
	}
	g = e + f;

	line5 = g;
    }
    if (0 == dependant)
    {
	print("Enter <1> if you are single, <0> if you are married: ");
	single = getinput();
	if (0 != single)
	{
	    line5 = 12950;
	}
	if (0 == single)
	{
	    line5 = 7200;
	}
    }

    line6 = line4 - line5;
    if (line6 < 0)
    {
	line6 = 0;
    }
    print("Your taxable income is: ");
    write(line6);

    print("Enter the amount of Federal income tax withheld: ");
    line7 = getinput();
    print("Enter <1> if you get an earned income credit (EIC); ");
    print("enter 0 otherwise: ");
    eic = getinput();
    line8 = 0;
    if (0 != eic)
    {
	print("OK, I'll give you a thousand dollars for your credit.\n");
	line8 = 1000;
    }
    line9 = line8 + line7;
    print("Your total tax payments amount to: ");
    write(line9);

    line10 = (line6 * 28 + 50) / 100;
    print("Your total tax liability is: ");
    write(line10);

    line11 = line9 - line10;
    if (line11 < 0)
    {
	line11 = 0;
    }
    if (line11 > 0)
    {
	print("Congratulations, you get a tax refund of $");
	write(line11);
    }

    line12 = line10-line9;
    if (line12 >= 0)
    {
	print("Bummer. You owe the IRS a check for $");
	write(line12);
    }
    if (line12 < 0)
    {
	line12 = 0;
    }

    print("Thank you for using ez-tax.\n");
}

"#;
static OUTPUT: &'static str = r#"#include <stdio.h>
#define read(x) scanf("%d",&x)
#define write(x) printf("%d\n",x)
#define print(x) printf(x)

int cse141getinput(void)
{
    int cse141a;
    cse141a = -1;
    while (0 > cse141a)
    {
	read(cse141a);
	if (0 > cse141a)
	{
	    print("I need a non-negative number: ");
	}
    }

    return cse141a;
}

int main() 
{
    int cse141line1, cse141line2, cse141line3, cse141line4, cse141line5, cse141line6, cse141line7, cse141line8, cse141line9, 
	cse141line10, cse141line11, cse141line12, cse141dependant, cse141single, cse141a, cse141b, cse141c, cse141d, cse141e, cse141f, cse141g, 
	cse141eic, cse141spousedependant;

    print("Welcome to the United States 1040 federal income tax program.\n");
    print("(Note: this isn't the real 1040 form. If you try to submit your\n");
    print("taxes this way, you'll get what you deserve!\n\n");

    print("Answer the following questions to determine what you owe.\n\n");

    print("Total wages, salary, and tips? ");
    cse141line1 = cse141getinput();
    print("Taxable interest (such as from bank accounts)? ");
    cse141line2 = cse141getinput();
    print("Unemployment compensation, qualified state tuition, and Alaska\n");
    print("Permanent Fund dividends? ");
    cse141line3 = cse141getinput();
    cse141line4 = cse141line1+cse141line2+cse141line3;
    print("Your adjusted gross income is: ");
    write(cse141line4);

    print("Enter <1> if your parents or someone else can claim you on their");
    print(" return. \nEnter <0> otherwise: ");
    cse141dependant = cse141getinput();
    if (0 != cse141dependant)
    {
	cse141a = cse141line1 + 250;
	cse141b = 700;
	cse141c = cse141b;
	if (cse141c < cse141a)
	{
	    cse141c = cse141a;
	}
	print("Enter <1> if you are single, <0> if you are married: ");
	cse141single = cse141getinput();
	if (0 != cse141single)
	{
	    cse141d = 7350;
	}
	if (0 == cse141single)
	{
	    cse141d = 4400;
	}
	cse141e = cse141c;
	if (cse141e > cse141d)
	{
	    cse141e = cse141d;
	}
	cse141f = 0;
	if (cse141single == 0)
	{
	    print("Enter <1> if your spouse can be claimed as a dependant, ");
	    print("enter <0> if not: ");
	    cse141spousedependant = cse141getinput();
	    if (0 == cse141spousedependant)
	    {
		cse141f = 2800;
	    }
	}
	cse141g = cse141e + cse141f;

	cse141line5 = cse141g;
    }
    if (0 == cse141dependant)
    {
	print("Enter <1> if you are single, <0> if you are married: ");
	cse141single = cse141getinput();
	if (0 != cse141single)
	{
	    cse141line5 = 12950;
	}
	if (0 == cse141single)
	{
	    cse141line5 = 7200;
	}
    }

    cse141line6 = cse141line4 - cse141line5;
    if (cse141line6 < 0)
    {
	cse141line6 = 0;
    }
    print("Your taxable income is: ");
    write(cse141line6);

    print("Enter the amount of Federal income tax withheld: ");
    cse141line7 = cse141getinput();
    print("Enter <1> if you get an earned income credit (EIC); ");
    print("enter 0 otherwise: ");
    cse141eic = cse141getinput();
    cse141line8 = 0;
    if (0 != cse141eic)
    {
	print("OK, I'll give you a thousand dollars for your credit.\n");
	cse141line8 = 1000;
    }
    cse141line9 = cse141line8 + cse141line7;
    print("Your total tax payments amount to: ");
    write(cse141line9);

    cse141line10 = (cse141line6 * 28 + 50) / 100;
    print("Your total tax liability is: ");
    write(cse141line10);

    cse141line11 = cse141line9 - cse141line10;
    if (cse141line11 < 0)
    {
	cse141line11 = 0;
    }
    if (cse141line11 > 0)
    {
	print("Congratulations, you get a tax refund of $");
	write(cse141line11);
    }

    cse141line12 = cse141line10-cse141line9;
    if (cse141line12 >= 0)
    {
	print("Bummer. You owe the IRS a check for $");
	write(cse141line12);
    }
    if (cse141line12 < 0)
    {
	cse141line12 = 0;
    }

    print("Thank you for using ez-tax.\n");
}

"#;

#[test]
fn test_scanner() {
    assert_str_eq!(OUTPUT, scan(INPUT))
}
