using BorderedMatrices
using LinearAlgebra, SparseArrays, Random, Test

@testset "BorderedVector" begin
    x = BorderedVector([1, 2, 3, 4, 5, 6], 7.0)
    @test length(x) == 7
    @test size(x) == (7, )
    @test size(x, 1) == 7
    @test size(x, 2) == 1
    @test nborder(x) == 1
    for i = 1:length(x)
        @test x[i] == i
    end
    @test collect(x) == x == [1, 2, 3, 4, 5, 6, 7]
    @test_throws BoundsError x[0]
    @test_throws BoundsError x[8]
    @test eltype(x) == Int

    x[1] = 2
    @test x[1] == 2
    x[7] = 9
    @test x._2 == 9
    @test_throws BoundsError x[0] = 1
    @test_throws BoundsError x[8] = 1

    for n = 5:20
        x = BorderedVector(rand(n), rand())
        @test size(x) == (n+1, )
        @test eltype(x) == Float64
        @test_throws BoundsError x[n+2]
    end

    # fancy indexing
    x = BorderedVector([1, 2, 3, 4], 5)
    x[1:end] .= 0
    @test x == [0, 0, 0, 0, 0]
    x = BorderedVector([1, 2, 3, 4], 5)
    x[[1, 5]] .= 0
    @test x == [0, 2, 3, 4, 0]
    x = BorderedVector([1, 2, 3, 4], 5)
    @test x[:] == [1, 2, 3, 4, 5]
    @test x[1:3] == [1, 2, 3]
    @test x[4:end] == [4, 5]

    # block border
    x = BorderedVector([1.0, 2, 3], [4, 5])
    @test nborder(x) == 2
    @test x == [1, 2, 3, 4, 5]
    x[5] = 6
    @test x._2 == [4, 6]

    # copy/similar
    x = BorderedVector([1, 2, 3, 4, 5, 6], 7)
    y = copy(x)
    y[1] = 5
    @test x[1] == 1
    s = similar(x)
    @test s isa BorderedVector
    @test size(s) == size(x) && eltype(s) == eltype(x)
end

@testset "BorderedMatrix" begin
    A = [1 2 3 1;
         2 3 4 1;
         2 3 4 1;
         1 2 3 1]
    b = [1, 2, 3, 4]
    c = [4, 3, 2, 1]
    M = BorderedMatrix(A, b, c, 0.0)
    @test size(M) == (5, 5)
    @test nborder(M) == 1
    @test Matrix(M) == [1  2  3  1  1;
                        2  3  4  1  2;
                        2  3  4  1  3;
                        1  2  3  1  4;
                        4  3  2  1  0] == M
    @test_throws BoundsError M[0, 0]
    @test_throws BoundsError M[1, 6]
    @test_throws BoundsError M[6, 6]
    @test eltype(M) == Int
    for i = 1:5, j = 1:5
        M[i, j] = i*j
    end
    @test M == [i*j for i = 1:5, j = 1:5]

    # copy does not alias
    N = copy(M)
    N._11[1, 1] = 5
    @test M[1, 1] == 1
    @test similar(M) isa BorderedMatrix

    # block borders, and partitioning a matrix
    K = rand(7, 7)
    M = BorderedMatrix(K, 2)
    @test nborder(M) == 2
    @test Matrix(M) == K == M
    for i = 1:7, j = 1:7
        M[i, j] = 10i + j
    end
    @test M == [10i + j for i = 1:7, j = 1:7]
    @test_throws DimensionMismatch BorderedMatrix(rand(3, 3), rand(3, 2), rand(3, 2), 1.0)
    @test_throws DimensionMismatch BorderedMatrix(rand(3, 3), rand(3, 2), rand(3, 2), rand(3, 3))
end

@testset "solves, single border" begin
    Random.seed!(0)
    for n = 10:5:100, alg in (:BEM, :BED), sp in (false, true)
        A = sp ? sprand(n, n, 0.5) + 10I : rand(n, n)
        M = BorderedMatrix(A, rand(n), rand(n), rand())
        r = BorderedVector(rand(n), rand())
        z = Matrix(M) \ collect(r)
        # factorise once, solve twice (the factorisation must not be modified by a solve)
        F = lu(M; alg = alg)
        @test F \ r ≈ z
        @test F \ r ≈ z
        @test F \ collect(r) ≈ z
        # original interface
        @test ldiv!(copy(M), copy(r), alg, true) ≈ z
        @test ldiv!(copy(M), copy(r), alg, false) ≈ z
    end
end

@testset "solves, block border" begin
    Random.seed!(1)
    for n = (20, 57), p = (1, 2, 5), alg in (:BEM, :BED)
        K = rand(n + p, n + p) + n*I
        M = BorderedMatrix(K, p)
        r = rand(n + p)
        F = lu(M; alg = alg)
        @test F \ r ≈ K \ r
        x = BorderedVector(r[1:n], r[n+1:end])
        @test F \ x ≈ K \ r
        @test M * (F \ x) ≈ x
    end
end

@testset "BEM is stable for nearly singular A" begin
    A = Float64[1 0     0;
                0 1     0;
                0 0 1e-32]
    M = BorderedMatrix(A, Float64[1, 1, 1], Float64[2, 2, 2], 1.0)
    r = BorderedVector([3.0, 4.0, 5.0], 1.0)
    @test cond(Matrix(M)) < 11
    @test cond(A) > 1e31
    x    = Matrix(M) \ collect(r)
    xBEM = lu(M; alg = :BEM) \ r
    xBED = lu(M; alg = :BED) \ r
    @test xBEM ≈ x
    @test !(xBED ≈ x)
end

@testset "BandedLU" begin
    Random.seed!(2)
    for n = (10, 100), (kl, ku) in ((0, 0), (1, 1), (3, 1), (2, 5))
        A = spdiagm([k => rand(n - abs(k)) for k in -kl:ku]...) + 4I
        @test bandwidths(A) == (kl, ku)
        F = bandedlu(A)
        b = rand(n)
        @test F \ b ≈ A \ b
        @test transpose(F) \ b ≈ transpose(Matrix(A)) \ b
        @test adjoint(F) \ b ≈ adjoint(Matrix(A)) \ b
        B = rand(n, 3)
        @test F \ B ≈ Matrix(A) \ B
        @test size(F) == (n, n)
    end
    # complex
    A = spdiagm(-1 => rand(ComplexF64, 9), 0 => rand(ComplexF64, 10) .+ 4, 2 => rand(ComplexF64, 8))
    F = bandedlu(A); b = rand(ComplexF64, 10)
    @test F \ b ≈ A \ b
    @test transpose(F) \ b ≈ transpose(Matrix(A)) \ b
    @test adjoint(F) \ b ≈ adjoint(Matrix(A)) \ b
    # explicit bandwidths larger than needed
    A = spdiagm(0 => rand(6) .+ 2, 1 => rand(5))
    @test bandedlu(A; kl = 2, ku = 3) \ ones(6) ≈ A \ ones(6)
    @test_throws ArgumentError bandedlu(A; kl = 0, ku = 0)
end

@testset "cyclic block-banded matrix as banded + border" begin
    # periodic block-banded matrix with half-bandwidth h blocks, plus one extra
    # dense row and column (as a phase condition and a period column)
    Random.seed!(3)
    r, Ns, h = 3, 20, 2
    n = r*Ns
    K = spzeros(n + 1, n + 1)
    for l in 1:Ns, j in -h:h
        m = mod(l - 1 + j, Ns) + 1
        K[(l-1)*r+1:l*r, (m-1)*r+1:m*r] = rand(r, r) + (j == 0 ? 4r*I : 0I)
    end
    K[1:n, n+1] = rand(n); K[n+1, 1:n] = rand(n)
    # border: the last h slices and the extra unknown
    p = h*r + 1
    M = BorderedMatrix(K, p)
    kl, ku = bandwidths(M._11)
    @test kl < (h + 1)*r && ku < (h + 1)*r      # the wrap-around is all in the border
    for alg in (:BEM, :BED)
        F = lu(M; alg = alg, factorize = bandedlu)
        b = rand(n + 1)
        @test F \ b ≈ Matrix(K) \ b
    end
end

@testset "complex and mixed element types" begin
    Random.seed!(5)
    n = 30
    for p in (1, 3), alg in (:BEM, :BED), fac in (lu, bandedlu)
        A = spdiagm(-2 => rand(ComplexF64, n-2), 0 => rand(ComplexF64, n) .+ 10, 1 => rand(ComplexF64, n-1))
        B = p == 1 ? rand(ComplexF64, n) : rand(ComplexF64, n, p)
        C = p == 1 ? rand(ComplexF64, n) : rand(ComplexF64, n, p)
        D = p == 1 ? rand(ComplexF64) : rand(ComplexF64, p, p)
        M = BorderedMatrix(A, B, C, D)
        K = Matrix(M)
        # the bottom block is the transpose of C, not its conjugate transpose
        @test K[n+1:end, 1:n] == transpose(reshape(C, n, :))
        F = lu(M; alg = alg, factorize = fac)
        r = rand(ComplexF64, n + p)
        @test F \ r ≈ K \ r
        @test F \ real(r) ≈ K \ real(r)                       # real right-hand side
        @test_throws ArgumentError ldiv!(F, real(r))           # cannot hold a complex solution
    end
    # real matrix, complex right-hand side
    for fac in (lu, bandedlu)
        A = spdiagm(-1 => rand(n-1), 0 => rand(n) .+ 4, 2 => rand(n-2))
        M = BorderedMatrix(A, rand(n), rand(n), 1.0)
        F = lu(M; factorize = fac)
        r = rand(ComplexF64, n + 1)
        @test F \ r ≈ Matrix(M) \ r
        s = copy(r); ldiv!(F, s)
        @test s ≈ Matrix(M) \ r
        x = BorderedVector(r[1:n], r[n+1])
        z = F \ x
        @test z isa BorderedVector{ComplexF64}
        @test z ≈ Matrix(M) \ r
    end
    # mixed element types are promoted
    M = BorderedMatrix(rand(4, 4) + 4I, rand(ComplexF64, 4), rand(4), 1.0)
    @test eltype(M) == ComplexF64
    @test lu(M) \ ones(5) ≈ Matrix(M) \ ones(5)
    @test eltype(BorderedVector(rand(3), 1im)) == ComplexF64
    M = BorderedMatrix(rand(4, 4) + 4I, rand(4), rand(4), 2im)
    @test eltype(M) == ComplexF64 && M[5, 5] == 2im
    v = rand(3); x = BorderedVector(v, 1)
    @test eltype(x) == Float64 && x._1 === v                  # same type: no copy
    @test eltype(BorderedVector([1, 2], [3.0])) == Int        # border converted to the main type
    # banded LU, real factorisation with complex right-hand sides
    A = spdiagm(0 => rand(6) .+ 3, 1 => rand(5)); F = bandedlu(A)
    b = rand(ComplexF64, 6); Bm = rand(ComplexF64, 6, 2)
    @test F \ b ≈ A \ b
    @test F \ Bm ≈ Matrix(A) \ Bm
    @test transpose(F) \ b ≈ transpose(Matrix(A)) \ b
    @test adjoint(F) \ b ≈ adjoint(Matrix(A)) \ b
end
