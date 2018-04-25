function [ distances ] = euclideandist( centers, radii )
%Function calculates distances between n number of points using the
%coordinates of the points as function inputs.
%   The distance formula is used to calculate all possible distances
%   between the n points
number_points = length(centers);
number_lines = number_points*(number_points-1)/2;
distances = zeros(1, number_lines);
k = 1;

if number_lines > 0
    for j = 1:(number_points - 1)
        for i = 1:(number_points - j)
            distances(k) = sqrt((centers(1,j) - centers(1,j+i))^2 + (centers(2,j) - centers(2,j+i))^2);
            k = k + 1;
        end
    end
else
    disp('Too few points to calculate distance');
end
end