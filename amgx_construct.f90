SUBROUTINE amgx_construct_matrix(x_min,             &
                           x_max,             &
                           y_min,             &
                           y_max,             &
                           nnz, &
                           rx, ry, &
                           b_mesh, &
                           b_rhs, &
                           x, &
                           Kx,           &
                           Ky,            &
                           A_cols, A_rows, A_data) bind(C, name="amgx_construct_matrix")

  USE, INTRINSIC::ISO_C_BINDING

  IMPLICIT NONE

  ! TODO make sure this works with PGI/Cray?
  INTEGER(C_INT):: x_min,x_max,y_min,y_max, nnz
  REAL(C_DOUBLE), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: Kx, Ky, b_mesh
  REAL(C_DOUBLE), DIMENSION(x_max*y_max) :: b_rhs, x
  REAL(C_DOUBLE) :: rx, ry

  INTEGER(C_INT), DIMENSION(x_max*y_max+1) ::  A_rows
  INTEGER(C_INT), DIMENSION(x_max*y_max*5) ::  A_cols
  REAL(C_DOUBLE), DIMENSION(x_max*y_max*5) ::  A_data

  INTEGER :: ctr, j, k, idx
  REAL(KIND=8) :: left, right, up, down, centre

  !
  !  |      | 4 |  
  !  |   ---+---+--
  !  |    1 | 2 | 3
  ! k|   ---+---+--
  !  |      | 0 |
  !  |
  !  +-------------
  !     j
  !
  ! 0    1 2 3     4
  !   0    1 2 3     4
  !     0    1 2 3     4
  ! etc
  !

  nnz = 0
  ctr = 1

  DO k=y_min, y_max
    DO j=x_min, x_max
      ctr = (x_max)*(k-1) + j
      A_rows(ctr) = nnz

      ! 0
      if (k .gt. y_min) then
        nnz = nnz + 1
        A_cols(nnz) = ctr - x_max
      endif
      ! 1
      if (j .gt. x_min) then
        nnz = nnz + 1
        A_cols(nnz) = ctr - 1
      endif

      ! 2
      nnz = nnz + 1
      A_cols(nnz) = ctr

      ! 3
      if (j .lt. x_max) then
        nnz = nnz + 1
        A_cols(nnz) = ctr + 1
      endif
      ! 4
      if (k .lt. y_max) then
        nnz = nnz + 1
        A_cols(nnz) = ctr + x_max
      endif

      ctr = ctr + 1
    ENDDO
  ENDDO

  a_rows(ctr) = nnz

  ! construct matrix
!$OMP PARALLEL PRIVATE(j, idx, left, right, up, down, centre, ctr)
!$OMP DO
  DO k=y_min, y_max
    DO j=x_min, x_max
      left = -Kx(j, k)*rx
      right = -Kx(j+1, k)*rx
      down = -Ky(j, k)*ry
      up = -Ky(j, k+1)*ry

      if (k .le. y_min) then
          down = 0.0_8
      endif
      if (j .le. x_min) then
          left = 0.0_8
      endif
      if (j .ge. x_max) then
          right = 0.0_8
      endif
      if (k .ge. y_max) then
          up = 0.0_8
      endif

      centre = 1.0_8 - left - right - up - down

      ! need to index sort of like C
      ! #define FTNREF2D(i_index,j_index,i_size,i_lb,j_lb)
      !   ((i_size)*(j_index-(j_lb))+(i_index)-(i_lb))
      ! x_host[ctr] = b_mesh[FTNREF2D(j  ,k  ,x_max+4,x_min-2,y_min-2)];
      idx = (x_max)*(k-1) + j
      idx = A_rows(idx)

      ctr = 0

      ! 0
      if (k .gt. y_min) then
        ctr = ctr + 1
        A_data(idx + ctr) = down
      endif
      ! 1
      if (j .gt. x_min) then
        ctr = ctr + 1
        A_data(idx + ctr) = left
      endif

      ! 2
      ctr = ctr + 1
      A_data(idx + ctr) = centre

      ! 3
      if (j .lt. x_max) then
        ctr = ctr + 1
        A_data(idx + ctr) = right
      endif
      ! 4
      if (k .lt. y_max) then
        ctr = ctr + 1
        A_data(idx + ctr) = up
      endif
    ENDDO
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

END SUBROUTINE

SUBROUTINE amgx_read_mesh(x_min,             &
                           x_max,             &
                           y_min,             &
                           y_max,             &
                           b_mesh, &
                           b_rhs, &
                           x) bind(C, name="amgx_read_mesh")

  USE, INTRINSIC::ISO_C_BINDING

  IMPLICIT NONE

  ! TODO make sure this works with PGI/Cray?
  INTEGER(C_INT):: x_min,x_max,y_min,y_max, nnz
  REAL(C_DOUBLE), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: b_mesh
  REAL(C_DOUBLE), DIMENSION(x_max*y_max) :: b_rhs, x

  INTEGER :: ctr, j, k, idx

!$OMP PARALLEL PRIVATE(ctr)
!$OMP DO
  DO k=y_min, y_max
    DO j=x_min, x_max
      ctr = (x_max)*(k-1) + j
      x(ctr) = b_mesh(j, k)
      b_rhs(ctr) = b_mesh(j, k)
    ENDDO
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

end subroutine

SUBROUTINE amgx_writeback(x_min,             &
                           x_max,             &
                           y_min,             &
                           y_max,             &
                           b_mesh, &
                           x) bind(C, name="amgx_writeback")

  USE, INTRINSIC::ISO_C_BINDING

  IMPLICIT NONE

  INTEGER(C_INT):: x_min,x_max,y_min,y_max
  REAL(C_DOUBLE), DIMENSION(x_min-2:x_max+2,y_min-2:y_max+2) :: b_mesh
  REAL(C_DOUBLE), DIMENSION(x_max*y_max) :: x

  INTEGER :: ctr, j, k

  ! write back
!$OMP PARALLEL PRIVATE(ctr)
!$OMP DO
  DO k=y_min, y_max
    DO j=x_min, x_max
      ctr = (x_max)*(k-1) + j
      b_mesh(j, k) = x(ctr)
    ENDDO
  ENDDO
!$OMP END DO
!$OMP END PARALLEL

end subroutine
