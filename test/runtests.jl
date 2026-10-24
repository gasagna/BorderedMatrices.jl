using BorderedMatrices
using LinearAlgebra, SparseArrays, Random, Test
using BandedMatrices: BandedMatrix, bandwidths

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
banded(T, n)  = BandedMatrix(spdiagm(-2 => rand(T, n-2), 0 => rand(T, n) .+ 10, 3 => rand(T, n-3)))

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

@testset "cyclic block-banded matrix: banded A, last slices in the border" begin
    # Ns slices of r unknowns, coupled to h slices on each side (periodically), plus one
    # extra unknown with a dense row and column
    Random.seed!(3)
    r, Ns, h = 3, 20, 2
    N = r*Ns + 1
    K = zeros(N, N)
    for l in 1:Ns, j in -h:h
        m = mod(l - 1 + j, Ns) + 1
        K[(l-1)*r+1:l*r, (m-1)*r+1:m*r] = rand(r, r) + (j == 0 ? 4r*I : 0I)
    end
    K[1:N-1, N] = rand(N - 1); K[N, 1:N-1] = rand(N - 1)
    # border: the last h slices and the extra unknown
    n = N - (h*r + 1)
    A = BandedMatrix(sparse(K[1:n, 1:n]))
    Ks = sparse(K)                                           # sparse borders: mostly zero
    M = BorderedMatrix(A, Ks[1:n, n+1:N], sparse(transpose(Ks[n+1:N, 1:n])), K[n+1:N, n+1:N])
    @test Matrix(M) == K
    @test all(bandwidths(A) .< (h + 1)*r)                    # the wrap-around is in the border
    b = rand(N)
    for alg in (:BEM, :BED)
        z = lu(M; alg = alg) \ BorderedVector(b[1:n], b[n+1:N])
        @test z ≈ K \ b
    end
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

@testset "blocked banded solves (BandedMatrices extension)" begin
    ext = Base.get_extension(BorderedMatrices, :BorderedMatricesBandedMatricesExt)
    @test ext !== nothing
    Random.seed!(6)
    for T in (Float64, ComplexF64), n in (1, 5, 63, 64, 65, 300), (kl, ku) in ((0, 0), (1, 0), (0, 2), (3, 2), (17, 40), (70, 70)),
        m in (1, 7)
        F = lu(BandedMatrix(rand(T, n, n), (kl, ku)))    # not diagonally dominant: rows are interchanged
        X = rand(T, n, m)
        @test ext.blocked_solve!(F, copy(X)) ≈ ldiv!(F, copy(X))
        @test ext.blocked_tsolve!(F, copy(X)) ≈ ldiv!(transpose(F), copy(X))
    end
end
