

clc; clear all;
screen = imread("screen.png");
aisan = imread("aisan.jpg");
aisan2 = imread("aisan2.png");
web = imread("web.jpg");

[row, col, d] = size(screen);
blank = ones(row, col, 3);

for i= 1:row
    for j=1:col
        blank(i, j, 1) = 0.5;
        blank(i, j, 2) = 0.5;
        blank(i, j, 3) = 0.5;
    end
end



mid = col/2;
left = col/12;
right = col*11/12;

line1 = row/30;


blank(:, mid-5:mid+5, 1) = 0;
blank(:, mid-5:mid+5, 2) = 1;
blank(:, mid-5:mid+5, 3) = 0;

blank(:, left-5:left+5, 1) = 0;
blank(:, left-5:left+5, 2) = 1;
blank(:, left-5:left+5, 3) = 0;

blank(:, right-5:right+5, 1) = 0;
blank(:, right-5:right+5, 2) = 1;
blank(:, right-5:right+5, 3) = 0;

blank(:, right-5:right+5, 1) = 0;
blank(:, right-5:right+5, 2) = 1;
blank(:, right-5:right+5, 3) = 0;

blank = drawH (blank, line1*1);
blank = drawH (blank, line1*5);
blank = drawH (blank, line1*10);
blank = drawH (blank, line1*15);
blank = drawH (blank, line1*20);
blank = drawH (blank, line1*25);
blank = drawH (blank, line1*29);

imtool(blank);

imwrite(blank, "blank.png");

%% 
function blank = drawH(I, line)
blank = I;

blank(line-5:line+5,:, 1) = 0;
blank(line-5:line+5,:, 2) = 1;
blank(line-5:line+5,:, 3) = 0;


end