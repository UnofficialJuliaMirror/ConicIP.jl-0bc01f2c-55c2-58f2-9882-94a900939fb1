module TestFactCheck

using ConicIP

import Base:*,+,\

using FactCheck
using Base.Test

FactCheck.setstyle(:default)

global tol    = 1e-4
global optTol = 1e-6

facts("ConicIP module") do

  srand(0)

  P_box(t,x) = [sign(xi)*(abs(xi) <= t ? abs(xi) : t) for xi in x];
  optcond(x, P, ∇f) = norm(x - P(x - ∇f(x)))/length(x);

  function compare(s1, s2::Dict)

    # Code for comparing two statuses (returned by Intpoint)

    return (s1.status == s2[:status] &&
            abs(s1.prFeas - s2[:prFeas]) < tol &&
            abs(s1.Mu - s2[:Mu]) < tol &&
            abs(s1.muFeas - s2[:muFeas]) < tol &&
            abs(s1.duFeas - s2[:duFeas]) < tol &&
            s1.Iter == s2[:Iter])
            
  end

  context("Box Constrained QP, H = I") do

    n = 1000;
    H = 0.5*Id(n)
    c = (1:n)''

    A = [speye(n); -speye(n)];
    b = -ones(2*n,1); k = size(A,1);
    ∇f = x -> H*(x - c);

    function kktsolver_2x2(Q,A,G)

      function solve2x2gen(F, F⁻¹)
        v = inv(F[1]*F[1]).diag
        D = Diagonal(v[1:n] + v[n+1:end])
        invHD = inv(ConicIP.Diag(H.diag + D.diag));
        return (rhs, rhs2) -> (invHD*rhs, zeros(0,1));
      end
      return solve2x2gen
      
    end

    sol = conicIP(H,H*c,A,b,[("R",2*n)],
                   kktsolver = pivot(kktsolver_2x2),
                   optTol = optTol,
                   DTB = 0.01,
                   maxRefinementSteps = 3);

    ystar = sol.y

    @fact optcond(ystar, x -> P_box(1,x), ∇f) --> less_than(tol);

    # These tests are a bit paranoid, but they make sure the solver doesn't
    # get changed in subtle ways which affects convergence.

    s = Dict(:status => :Optimal,
             :prFeas => 2.3279184881643e-16,
             :Mu => 0.0014841398479756253,
             :muFeas => 3.5147719126896054e-7,
             :duFeas => 8.841314541553926e-8,
             :Iter => 7)

    @fact compare(sol,s) --> true

  end

  for kktsolver = (ConicIP.kktsolver_qr, 
                   ConicIP.kktsolver_sparse, 
                   pivot(ConicIP.kktsolver_2x2))

    context("Projection onto Sphere") do

      # minimize (1/2)‖x - a‖² subject to ‖x|‖₂ < 1
      #
      # (transformed to)
      #
      # minimize (0.5)x'*H*x + c'*x subject to (0,x)' <= -e1

      n = 2;
      H = Id(n);
      a = ones(n,1);
      A = [spzeros(1,n); speye(n)];
      b = [-1;zeros(n,1)];

      sol = conicIP(H,H*a,A,b,[("Q",n+1)],
                    optTol = optTol,
                    DTB = 0.01,
                    kktsolver = kktsolver,
                    maxRefinementSteps = 3);

      ystar = sol.y

      @fact norm(ystar - a/norm(a)) --> less_than(tol);

      # These tests are a bit paranoid, but they make sure the solver doesn't
      # get changed in subtle ways which affects convergence.

      s = Dict(:status =>:Optimal,
               :prFeas => 0.0,
               :Mu => 2.866608128093695e-7,
               :muFeas => 1.621702501927476e-7,
               :duFeas => 3.2367552452111847e-16,
               :Iter => 5)

      @fact compare(sol, s) --> true

    end

    context("Combined") do

      n = 10;
      H = Id(n);
      c = (1:n)'';

      A = [ speye(n)      ; # R
            spzeros(1,n)  ; # Q
            speye(n)     ]; #

      b = [zeros(n,1) ;
           -1.        ;
           zeros(n,1)];

      sol = conicIP(H,H*c,A,b,[("R",n),("Q",n+1)],
                    optTol = optTol,
                    DTB = 0.01,
                    kktsolver = kktsolver,                     
                    maxRefinementSteps = 3);

      ystar = sol.y

      y = [max(0,i) for i in c];
      y = y/norm(y);

      @fact norm(ystar - y) --> less_than(tol);

      # These tests are a bit paranoid, but they make sure the solver doesn't
      # get changed in subtle ways which affects convergence.

      s = Dict(:status => :Optimal,
               :prFeas => 7.764421906286858e-17,
               :Mu => 4.663886012743681e-7,
               :muFeas => 1.7037397157416066e-7,
               :duFeas => 2.77947804665922e-17,
               :Iter => 10)

      @fact compare(sol, s) --> true

    end

    context("Projection onto simplex (Linear Equality Test)") do

      n = 10;
      H = eye(n)
      c = (1:n)''

      A = speye(n)
      b = zeros(n,1);
      k = size(A,1)

      G = ones(1,n)
      d = ones(1,1)

      sol = conicIP(H,H*c,
                    A,b,[("R",n)],
                    G,d,
                    kktsolver = kktsolver,                              
                    optTol = optTol/100);

      ystar = sol.y

      ysol  = zeros(n,1); ysol[n] = 1;

      @fact norm(ystar - ysol) --> less_than(tol)

      # These tests are a bit paranoid, but they make sure the solver doesn't
      # get changed in subtle ways which affects convergence.

      s = Dict(:status => :Optimal,
               :prFeas => 1.4506364239112378e-16,
               :Mu => 2.7686402945528533e-9,
               :muFeas => 2.897827518851058e-9,
               :duFeas => 2.70780035221441e-17,
               :Iter => 11)

      @fact compare(sol, s) --> true

    end

    context("Projection onto simplex, dense H") do

      n = 10;
      H = randn(n); H = H*H'
      c = (1:n)''

      A = speye(n)
      b = zeros(n,1);
      k = size(A,1)

      G = ones(1,n)
      d = ones(1,1)

      sol = ConicIP.conicIP(H,H*c,
                            A,b,[("R",n)],
                            G,d,
                            kktsolver = kktsolver,
                            optTol = optTol/100);

      ystar = sol.y

      if kktsolver == ConicIP.kktsolver_sparse
        println("Pivot")
        s = Dict(:status => :Optimal,
                 :prFeas => 4.488229069360946e-16,
                 :Mu => 2.1436595135398927e-8,
                 :muFeas => 3.000777220457259e-9,
                 :duFeas => 6.279962324264275e-17,
                 :Iter => 7)
      else
        s = Dict(:status => :Optimal,
                 :prFeas => 4.488229069360946e-16,
                 :Mu => 2.1436595135398927e-8,
                 :muFeas => 3.000777220457259e-9,
                 :duFeas => 6.279962324264275e-17,
                 :Iter => 8)
      end
      @fact compare(sol, s) --> true

    end

    context("Projection onto simplex, dense H, Random Projection") do

      n = 10;
      H = randn(n); H = H*H'
      c = (1:n)''

      A = speye(n)
      b = zeros(n,1);
      k = size(A,1)

      G = rand(6,n)
      d = zeros(6,1)

      ystar = conicIP(H,H*c,
                      A,b,[("R",n)],
                      G,d,
                      kktsolver = kktsolver,
                      optTol = optTol/100).y;

      ysol  = zeros(n,1); ysol[n] = 1;

    end

    context("Linear Constraints Comparison") do

      n = 10;
      H = randn(n); H = H*H'
      c = (1:n)''

      A = speye(n)
      b = zeros(n,1);
      k = size(A,1)

      G = rand(6,n)
      d = zeros(6,1)

      ystar1 = conicIP(H,H*c,
                       A,b,[("R",n)],
                       G,d,
                       kktsolver = kktsolver,              
                       optTol = optTol/100).y;

      ystar2 = conicIP(H,H*c,
                       [A; G; -G],[b; d; -d],[("R",(n + 2*6))],
                       G,d,
                       optTol = optTol/100).y;

      @fact norm(ystar1 - ystar2) --> less_than(tol)

    end

    context("Infeasible") do

      n = 10;
      H = randn(n); H = H*H'
      c = (1:n)''

      A = [speye(n);-speye(n)]
      b = [ones(n,1); ones(n,1)]
      k = size(A,1)

      sol= conicIP(H,H*c,
                   A,b,[("R",2*n)],
                   kktsolver = kktsolver,                             
                   optTol = optTol/100);

      @fact sol.status --> :Infeasible

    end

    context("Infeasible (With linear constraints)") do

      n = 10;
      H = randn(n); H = H*H'
      c = (1:n)''

      A = speye(n)
      b = zeros(n,1)

      G = [1 zeros(1,9)]
      d = -ones(1,1)

      k = size(A,1)

      sol= conicIP(H,H*c,
                   A,b, [("R",n)],
                   G,d,
                   kktsolver = kktsolver,                             
                   optTol = optTol/100);

      @fact sol.status --> :Infeasible

    end  

    context("Unbounded") do

      n = 10;
      H = zeros(n,n);
      c = (1:n)''

      A = speye(n)
      b = zeros(n,1)

      sol = conicIP(H,c,
                    A,b,[("R",n)],
                    kktsolver = kktsolver,                              
                    optTol = optTol/100)

      @fact sol.status --> :DualInfeasible

    end

    context("Bad Input") do

      n = 10;
      H = zeros(n,n);
      c = (1:n)''

      A = speye(n+2)
      b = zeros(n,1)

      try
        sol = conicIP(H,c,
                      A,b,[("R",n)],
                      kktsolver = kktsolver,                                
                      optTol = optTol/100)
      catch
        @fact 1 --> 1
      end

    end

  end

  context("SDP - Projection onto PSD Matrix") do

    # Implement This

    n = 21;
    H = eye(n,n);
    c = ConicIP.vecm(diagm([1;1;1;-1;-1;-1]))''

    A = speye(21)
    b = zeros(21,1)

    sol = conicIP(H,c,
                  A,b,[("S",n)],
                  optTol = optTol/100)


    s = Dict(:status => :Optimal,
             :prFeas => 4.2341217602756234e-16,
             :Mu => 3.4583513329836624e-10,
             :muFeas => 1.48267911727847e-9,
             :duFeas => 4.2341217602756234e-16,
             :Iter => 6)

    @fact norm(ConicIP.mat(sol.y) - diagm([1;1;1;0;0;0]), Inf) --> less_than(tol)
    @fact compare(sol, s) --> true
            
  end

end

end # module
