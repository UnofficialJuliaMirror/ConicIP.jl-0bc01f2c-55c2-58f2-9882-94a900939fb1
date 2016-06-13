using IntPoint

n = 300000

Q = speye(n);
c = ones(n,1);
A = speye(n);
b = zeros(n,1);
𝐾 = [("R",n)];

#@time intpoint( Q , c , A , b , 𝐾 , verbose = true);

function solve2x2gen(F, F⁻¹_, Q_, A_, G_)
#function solve2x2gen(F)

  # return a function (y,x) ↦ ( (Q + diag(u))⁻¹y , zeros(0,1) )
  QpD⁻¹ = cholfact(Q + spdiagm( (F[1].diag).^(-2) ))
  return (y, x) -> (QpD⁻¹\y, zeros(0,1))

end

Profile.clear()

tic()
@profile intpoint( Q , c , A , b , 𝐾 , solve3x3gen = pivot3gen(solve2x2gen); verbose = true);	
toc()

ProfileView.view()

@time sol = intpoint( Q , c , A , b , 𝐾 , solve3x3gen = pivot3gen(solve2x2gen); verbose = true);	

n = 300000

Q = speye(n);
c = ones(n,1);
A = speye(n);
b = zeros(n,1);
𝐾 = [("R",n)];

function solve2x2gen(F)

  # return a function (y,x) ↦ ( (Q + diag(u))⁻¹y , zeros(0,1) )
  QpD⁻¹ = cholfact(Q + spdiagm( (F[1].diag).^(-2) ))
  return (y, x) -> (QpD⁻¹\y, zeros(0,1))

end

@time intpoint( Q , c , A , b , 𝐾 , solve2x2gen = solve2x2gen; verbose = true);	
