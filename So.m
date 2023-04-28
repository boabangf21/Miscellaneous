   function r = So(tau,h_state)
    % shrinkage operator
    r = sign(h_state) .* max(abs(h_state) - tau, 0);
     end