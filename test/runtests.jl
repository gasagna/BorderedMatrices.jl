using BorderedMatrices
using LinearAlgebra, SparseArrays, Random, Test

# a random bordered system with element type T, top left block built by `mkA`,
# dense or sparse borders
function system(mkA, T, n, p; sparseborders = false)
    B, C = sparseborders ? (sprand(T, n, p, 0.1), sprand(T, n, p, 0.1)) : (rand(T, n, p), rand(T, n, p))
    M = BorderedMatrix(mkA(T, n), B, C, rand(T, p, p))
    r = BorderedVector(rand(T, n), rand(T, p))
    return M, r
end
dense(T, n)   = rand(T, n, n) + n*I
sparse_(T, n) = sprand(T, n, n, 0.2) + n*I
function banded(T, n)
    A = BandedMatrix{T}(n, 2, 3)
    for j in 1:n, i in max(1, j - 3):min(n, j + 2)
        A[i, j] = rand(T) + (i == j ? 10 : 0)
    end
    return A
end

@testset "BorderedVector" begin
    v = BorderedVector([1.0, 2, 3], [4.0, 5])
    @test size(v) == (5,)
    @test v == [1, 2, 3, 4, 5]
    v[2] = 7; v[5] = 8
    @test v.x == [1, 7, 3] && v.y == [4, 8]
    @test_throws BoundsError v[6]
    w = copy(v); w[1] = 0
    @test v[1] == 1
    @test similar(v) isa BorderedVector{Float64}
    @test_throws MethodError BorderedVector([1.0], [1im])    # one element type
end

@testset "BorderedMatrix" begin
    A = [1 2; 3 4]; B = reshape([5, 6], 2, 1); C = reshape([7, 8], 2, 1); D = fill(9, 1, 1)
    M = BorderedMatrix(A, B, C, D)
    @test size(M) == (3, 3)
    @test M == [1 2 5; 3 4 6; 7 8 9]                         # the bottom block is Cᵀ
    @test_throws DimensionMismatch BorderedMatrix(A, B, C, fill(9, 2, 2))
    @test_throws MethodError BorderedMatrix(A, B, C, fill(9.0, 1, 1))   # one element type
    @test_throws MethodError BorderedMatrix(A, sparse(B), C, D)         # B and C of the same type
end

@testset "solves" begin
    Random.seed!(0)
    for T in (Float64, ComplexF64), mkA in (dense, sparse_, banded), p in (1, 3), alg in (:BEM, :BED),
        sparseborders in (false, true)
        M, r = system(mkA, T, 40, p; sparseborders)
        z = Matrix(M) \ collect(r)
        F = lu(M; alg = alg)
        @test F \ r ≈ z
        @test F \ r ≈ z                                       # a solve does not modify F
        s = copy(r); ldiv!(F, s)
        @test s ≈ z
    end
    # A factorised in place
    M, r = system(dense, Float64, 20, 2)
    z = Matrix(M) \ collect(r)
    @test lu(M; factorize = lu!) \ r ≈ z
    @test_throws ArgumentError lu(M; alg = :XYZ)
end

@testset "BEM is stable for nearly singular A" begin
    A = [1.0 0 0; 0 1 0; 0 0 1e-32]
    M = BorderedMatrix(A, ones(3, 1), fill(2.0, 3, 1), ones(1, 1))
    r = BorderedVector([3.0, 4, 5], [1.0])
    @test cond(Matrix(M)) < 11
    z = Matrix(M) \ collect(r)
    @test lu(M; alg = :BEM) \ r ≈ z
    @test !(lu(M; alg = :BED) \ r ≈ z)
end

@testset "blockcyclic" begin
    # Ns slices of r unknowns, coupled to h slices on each side (periodically), plus q
    # extra unknowns with dense rows and columns
    Random.seed!(3)
    for T in (Float64, ComplexF64), (r, Ns, h, q) in ((3, 20, 2, 1), (1, 7, 3, 2), (4, 9, 1, 1))
        blocks = Dict((l, mod(l - 1 + j, Ns) + 1) => rand(T, r, r) + (j == 0 ? 4r*I : 0I) for l in 1:Ns, j in -h:h)
        b, c, d = rand(T, r*Ns, q), rand(T, r*Ns, q), rand(T, q, q)
        K = zeros(T, r*Ns + q, r*Ns + q)
        for ((l, k), F) in blocks
            K[(l-1)*r+1:l*r, (k-1)*r+1:k*r] = F
        end
        K[1:r*Ns, r*Ns+1:end] = b; K[r*Ns+1:end, 1:r*Ns] = transpose(c); K[r*Ns+1:end, r*Ns+1:end] = d
        M = blockcyclic((l, k) -> blocks[(l, k)], r, Ns, h, b, c, d)
        @test M isa BorderedMatrix{T, BandedMatrix{T}}
        @test Matrix(M) == K                                 # same matrix, same ordering
        @test M.A.kl == M.A.ku == (h + 1)*r - 1
        x = rand(T, r*Ns + q); n = size(M.A, 1)
        for alg in (:BEM, :BED)
            @test collect(lu(M; alg = alg) \ BorderedVector(x[1:n], x[n+1:end])) ≈ K \ x
        end
    end
    @test_throws ArgumentError blockcyclic((l, k) -> ones(1, 1), 1, 4, 2, ones(4, 1), ones(4, 1), ones(1, 1))
end

allocs(F, s) = @allocated ldiv!(F, s)

@testset "solves do not allocate" begin
    for mkA in (dense, banded), alg in (:BEM, :BED), sparseborders in (false, true)
        M, r = system(mkA, Float64, 50, 3; sparseborders)
        F = lu(M; alg = alg)
        allocs(F, copy(r))                           # compile
        @test allocs(F, copy(r)) == 0
    end
end

@testset "BandedMatrix and its LU" begin
    Random.seed!(6)
    for T in (Float64, ComplexF64), n in (1, 5, 63, 64, 65, 300), (kl, ku) in ((0, 0), (1, 0), (0, 2), (3, 2), (17, 40), (70, 70)),
        m in (0, 1, 7)
        # a banded matrix, not diagonally dominant: rows are interchanged
        A = BandedMatrix{T}(n, kl, ku)
        for j in 1:n, i in max(1, j - ku):min(n, j + kl)
            A[i, j] = rand(T)
        end
        @test A == [i - j <= kl && j - i <= ku ? A[i, j] : zero(T) for i in 1:n, j in 1:n]
        F = lu(A)
        X = m == 0 ? rand(T, n) : rand(T, n, m)              # m = 0: a vector (gbtrs)
        @test F \ X ≈ A \ X
        @test transpose(F) \ X ≈ transpose(A) \ X
        @test ldiv!(F, copy(X)) ≈ A \ X
        # blocked and LAPACK solves agree, whatever the bandwidth
        if m > 0
            Y = copy(X); BorderedMatrices._lsolve!(F, Y); BorderedMatrices._usolve!(F, Y)
            @test Y ≈ LAPACK.gbtrs!('N', kl, ku, n, F.AB, F.ipiv, copy(X))
            Y = copy(X); BorderedMatrices._utsolve!(F, Y); BorderedMatrices._ltsolve!(F, Y)
            @test Y ≈ LAPACK.gbtrs!('T', kl, ku, n, F.AB, F.ipiv, copy(X))
        end
    end
    A = BandedMatrix{ComplexF64}(6, 5, 5); A .= rand(ComplexF64, 6, 6) + 6I
    F = lu(A); b = rand(ComplexF64, 6)
    @test adjoint(F) \ b ≈ adjoint(Matrix(A)) \ b
    # entries outside the band: zero, and only zero can be stored there
    A = BandedMatrix{Float64}(5, 1, 2)
    @test A[5, 1] == 0
    A[5, 1] = 0
    @test_throws ArgumentError A[5, 1] = 1
    @test_throws BoundsError A[6, 1]
    # lu! works in place, lu on a copy
    for i in 1:5
        A[i, i] = i + 1
    end
    A[1, 2] = 1; A[2, 1] = 1
    B = copy(A); lu(A)
    @test A == B
    lu!(A)
    @test A.AB != B.AB                                       # overwritten by the factors
end
