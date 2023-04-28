firstparam = [1, 2, 3.3, 3.7, 8, 21];  %list of places to search for first parameter
secondparam = linspace(0,1,20);        %list of places to search for second parameter
[F,S] = ndgrid(firstparam, secondparam);

