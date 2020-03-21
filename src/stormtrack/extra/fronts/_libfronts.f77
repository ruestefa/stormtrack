c     ----------------------------------------------------------------------
c     Subroutines for Frontal Detection Tool
c     ----------------------------------------------------------------------

c     -----------------------------------------------
c     label connected fronts
c     -----------------------------------------------

      subroutine clustering(outar,ntot,inar,nx,ny,grid)

c     Given the input array <inar> with 0/1 entries (for example the
c     NCOFF array), the individual events are counted and each event
c     gets an individual 'stamp' which is specific to this event. An
c     event is characterised by 1-entries in <inar> which are connected.
c     <nx>,<ny> give the dimension of the arrays, <ntot> the total
c     number of events i.e. objects.

      implicit none

c     Declaration of subroutine parameters
      integer, intent(in)   :: nx,ny
      integer, intent(in)   :: grid
      integer, intent(inout)  :: outar(nx,ny)
      integer, intent(out)  :: ntot
      real, intent(in)      :: inar(nx,ny)

c     Auxiliary variables
      integer     :: i,j

c     Copy inar to outar
      outar=nint(inar)

c     -------- Do the clustering -----------
c     total number of cluster
      ntot=1

c     Find an element which does not yet belong to a cluster
       do j=1,ny
        do i=1,nx
          if (outar(i,j).eq.1) then
            ntot=ntot+1
            outar(i,j)=ntot
            call connect(outar,ntot,i,j,nx,ny,grid)
          endif
        enddo
      enddo

c     Correct the output array (so far, the index is 1 unit too large)
      where (outar.gt.1) outar=outar-1
      ntot=ntot-1

c     -----------------------------------------------
      end subroutine clustering
c     -----------------------------------------------

c     -----------------------------------------------
c     mark connected gridpoints with fronts
c     -----------------------------------------------

      subroutine connect (outar,label,i,j,nx,ny,grid)

      implicit none

c     Mark all elements in outar which are connected to element i,j

c     Declaration of subroutine parameters
      integer, intent(in)   :: nx,ny
      integer, intent(in)   :: grid
      integer, intent(in)   :: label
      integer, intent(in)     :: i,j
      integer, intent(inout)  :: outar(nx,ny)

c     Auxiliary variables
      integer  :: il,ir,ju,jd,im,jm
      integer  :: stack
      integer  :: indx(nx*ny),indy(nx*ny)

c     Push the first element on the stack
      stack=1
      indx(stack)=i
      indy(stack)=j

c     Define the indices of the neighboring elements
      wstack: do

        il=indx(stack)-1
        ir=indx(stack)+1
        ju=indy(stack)+1
        jd=indy(stack)-1
        im=indx(stack)
        jm=indy(stack)
        stack=stack-1

c      Grid handling
c       periodic closed, cyclic point
        if ( grid.eq.2 ) then
           if ( il.eq.0      ) il = nx-1
           if ( ir.eq.(nx+1) ) ir = 2
c       periodic no cyclic point (not closed)
        elseif ( grid.eq.1 ) then
           if ( il.eq.0      ) il = nx
           if ( ir.eq.(nx+1) ) ir = 1
c       not periodic
        else
           if ( il.eq.0      ) il = 1
           if ( ir.eq.(nx+1) ) ir = nx
        endif
        if (ju.gt.ny) ju=ny
        if (jd.lt.1 ) jd=1

c       Mark all connected elements (build up the stack)
        if (outar(il,jm).eq.1) then
          outar(il,jm)=label
          stack=stack+1
          indx(stack)=il
          indy(stack)=jm
        endif
        if (outar(ir,jm).eq.1) then
          outar(ir,jm)=label
          stack=stack+1
          indx(stack)=ir
          indy(stack)=jm
        endif
        if (outar(im,ju).eq.1) then
          outar(im,ju)=label
          stack=stack+1
          indx(stack)=im
          indy(stack)=ju
        endif
        if (outar(im,jd).eq.1) then
          outar(im,jd)=label
          stack=stack+1
          indx(stack)=im
          indy(stack)=jd
        endif
        if (outar(il,jd).eq.1) then
          outar(il,jd)=label
          stack=stack+1
          indx(stack)=il
          indy(stack)=jd
        endif
        if (outar(ir,jd).eq.1) then
          outar(ir,jd)=label
          stack=stack+1
          indx(stack)=ir
          indy(stack)=jd
        endif
        if (outar(il,ju).eq.1) then
          outar(il,ju)=label
          stack=stack+1
          indx(stack)=il
          indy(stack)=ju
        endif
        if (outar(ir,ju).eq.1) then
          outar(ir,ju)=label
          stack=stack+1
          indx(stack)=ir
          indy(stack)=ju
        endif

        if (stack.eq.0) exit wstack

      enddo wstack

c     -----------------------------------------------
      end subroutine connect
c     -----------------------------------------------


c     -----------------------------------------------------------------
c     Gradient
c     -----------------------------------------------------------------

      subroutine grad(gradx,grady,scalar,
     >                vert,xmin,ymin,dx,dy,nx,ny,nz,stride,mdv)

c     Calculate the two vector components <gradx> and <grady> of the
c     scalar field <scalar>.  The vertical coordinate is specified in
c     <vert>, the grid in <xmin,ymin,dx,dy,nx,ny,nz,mdv>.

      implicit none

c     Declaration of subroutine parameters
      integer nx,ny,nz,stride
      real    gradx(nx,ny,nz)
      real    grady(nx,ny,nz)
      real    scalar(nx,ny,nz)
      real    vert(nx,ny,nz)
      real    xmin,ymin,dx,dy
      real    mdv

c     Calculate the derivatives in x and y direction
      call deriv (gradx,scalar,'x',vert,xmin,ymin,dx,dy,
     >            nx,ny,nz,stride,mdv)
      call deriv (grady,scalar,'y',vert,xmin,ymin,dx,dy,
     >            nx,ny,nz,stride,mdv)

      end

c     -----------------------------------------------------------------
c     Create the aura around a 3d field
c     -----------------------------------------------------------------

      subroutine aura (gri,aur,dir,nx,ny,nz,xmin,ymin,dx,dy,mdv)

c     Create a one-point aura around the grid, in order to avoid nasty
c     problems when calculating fields at boundary points

      implicit none

c     Declaration of subroutine parameters
      integer nx,ny,nz
      real    gri(nx,ny,nz)
      real    aur(0:nx+1,0:ny+1,0:nz+1)
      integer dir
      real    xmin,ymin,dx,dy
      real    mdv

c     Numerical and physical parameters
      real       eps
      parameter  (eps=0.01)

c     Auxiliary variables
      integer i,j,k
      real    xmax,ymax
      integer domx,domy
      real    mean,count
      integer kmin,kmax

c     Set the z dimension for the output array
      if (nz.gt.1) then
         kmin=0
         kmax=nz+1
      elseif (nz.eq.1) then
         kmin=0
         kmax=2
      endif

c     Determine the x topology of the grid
c     1: periodic, not closed;
c     2: periodic, closed;
c     0: not periodic (and therefore not closed)
      xmax=xmin+real(nx-1)*dx
      ymax=ymin+real(ny-1)*dy
      if (abs(xmax-xmin-360.).lt.eps) then
         domx=2
      elseif (abs(xmax-xmin-360.+dx).lt.eps) then
         domx=1
      else
         domx=0
      endif

c     Determine the y topology of the grid
c     1    : neither north, nor south pole;
c     mod 2: exactly at south pole (closed)
c     mod 3: exactly at north pole (closed)
c     mod 5: one grid point north of south pole (not closed)
c     mod 7: one grid point south of north pole (not closed)
      domy=1
      if (abs(ymin+90.).lt.eps) then
         domy=2
      endif
      if (abs(ymax-90.).lt.eps) then
         domy=domy*3
      endif
      if (abs(ymin+90.-dy).lt.eps) then
         domy=domy*5
      endif
      if (abs(ymin-90.+dy).lt.eps) then
         domy=domy*7
      endif

c     Forward transformation (create aura)
      if (dir.eq.1) then

c        Copy the interior part
         aur(1:nx,1:ny,1:nz)=gri(1:nx,1:ny,1:nz)

c        Upper boundary
         if (nz.gt.1) then
            do i=1,nx
               do j=1,ny
                  if ((abs(aur(i,j,  nz)-mdv).gt.eps).and.
     >                (abs(aur(i,j,nz-1)-mdv).gt.eps)) then
                     aur(i,j,nz+1) = 2.*aur(i,j,nz) - aur(i,j,nz-1)
                  else
                     aur(i,j,nz+1) = mdv
                  endif
               enddo
            enddo
         else
            do i=1,nx
               do j=1,ny
                  aur(i,j,2)=aur(i,j,1)
               enddo
            enddo
         endif

c        Lower boundary
         if (nz.gt.1) then
            do i=1,nx
               do j=1,ny
                  if ((abs(aur(i,j,1)-mdv).gt.eps).and.
     >                (abs(aur(i,j,2)-mdv).gt.eps)) then
                     aur(i,j,0) = 2.*aur(i,j,1) - aur(i,j,2)
                  else
                     aur(i,j,0) = mdv
                  endif
               enddo
            enddo
         else
            do i=1,nx
               do j=1,ny
                  aur(i,j,0)=aur(i,j,1)
               enddo
            enddo
         endif

c        Northern and southern boundary, not near the poles
         if (mod(domy,1).eq.0) then
            do i=1,nx
               do k=kmin,kmax
                  if ((abs(aur(i,  ny,k)-mdv).gt.eps).and.
     >                (abs(aur(i,ny-1,k)-mdv).gt.eps)) then
                     aur(i,ny+1,k) = 2.*aur(i,ny,k)-aur(i,ny-1,k)
                  else
                     aur(i,ny+1,k) = mdv
                  endif
                  if  ((abs(aur(i,1,k)-mdv).gt.eps).and.
     >                 (abs(aur(i,2,k)-mdv).gt.eps)) then
                     aur(i,0,k) = 2.*aur(i,1,k)-aur(i,2,k)
                  else
                     aur(i,0,k) = mdv
                  endif
               enddo
            enddo
         endif

c        Southern boundary, one grid point north of pole
c        Set the pole point to the mean of the nearest points
         if (mod(domy,5).eq.0) then
            do k=kmin,kmax
               mean=0.
               count=0.
               do i=1,nx
                  if (abs(aur(i,1,k)-mdv).gt.eps) then
                     mean=mean+aur(i,1,k)
                     count=count+1.
                  endif
               enddo
               if (count.gt.0.) then
                  mean=mean/count
               else
                  mean=mdv
               endif
               do i=1,nx
                  aur(i,0,k) = mean
               enddo
            enddo
         endif

c        Northern boundary, one grid point south of pole
c        Set the pole point to the mean of the nearest points
         if (mod(domy,7).eq.0) then
            do k=kmin,kmax
               mean=0.
               count=0.
               do i=1,nx
                  if (abs(aur(i,ny,k)-mdv).gt.eps) then
                     mean=mean+aur(i,ny,k)
                     count=count+1.
                  endif
               enddo
               if (count.gt.0.) then
                  mean=mean/count
               else
                  mean=mdv
               endif
               do i=1,nx
                  aur(i,ny+1,k) = mean
               enddo
            enddo
         endif

c        Southern boundary, exactly at south pole
         if (mod(domy,2).eq.0) then
            do i=1,nx
               do k=kmin,kmax
                  aur(i,0,k)=mdv
               enddo
            enddo
         endif

c        Northern boundary, exactly at north pole
         if (mod(domy,3).eq.0) then
            do i=1,nx
               do k=kmin,kmax
                  aur(i,ny+1,k)=mdv
               enddo
            enddo
         endif

c        The domain is periodic in x, but not closed (no cyclic point)
         if (domx.eq.1) then
            do j=0,ny+1
               do k=kmin,kmax
                  aur(   0,j,k) = aur(nx,j,k)
                  aur(nx+1,j,k) = aur( 1,j,k)
               enddo
            enddo
         endif

c        The domain is periodic in x and closed (cyclic point)
         if (domx.eq.2) then
            do j=0,ny+1
               do k=kmin,kmax
                  aur(   0,j,k) = aur(nx-1,j,k)
                  aur(nx+1,j,k) = aur(   2,j,k)
               enddo
            enddo
         endif

c        The domain is not periodic in x
         if (domx.eq.0) then
            do j=0,ny+1
               do k=kmin,kmax
                  if ((abs(aur(1,j,k)-mdv).gt.eps).and.
     >                (abs(aur(2,j,k)-mdv).gt.eps)) then
                     aur(0,j,k) = 2.*aur(1,j,k) - aur(2,j,k)
                  else
                     aur(0,j,k) = mdv
                  endif
                  if ((abs(aur(  nx,j,k)-mdv).gt.eps).and.
     >                (abs(aur(nx-1,j,k)-mdv).gt.eps)) then
                     aur(nx+1,j,k) = 2.*aur(nx,j,k) - aur(nx-1,j,k)
                  else
                     aur(nx+1,j,k) = mdv
                  endif
               enddo
            enddo
         endif

      endif

c     Backward transformation
      if (dir.eq.-1) then

         if (nz.gt.1) then
            gri(1:nx,1:ny,1:nz)=aur(1:nx,1:ny,1:nz)
         elseif (nz.eq.1) then
            gri(1:nx,1:ny,1)=aur(1:nx,1:ny,1)
         endif

      endif

      end

c     -----------------------------------------------------------------
c     Horizontal and vertical derivatives for 3d fields
c     -----------------------------------------------------------------

      subroutine deriv (dfield,field,direction,
     >                  vert,xmin,ymin,dx,dy,nx,ny,nz,stride,mdv)

c     Calculate horizontal and vertical derivatives of the 3d field <field>.
c     The direction of the derivative is specified in <direction>
c         'x','y'          : Horizontal derivative in x and y direction
c         'p','z','t','m'  : Vertical derivative (pressure, height, theta, model)
c     The 3d field <vert> specifies the isosurfaces along which the horizontal
c     derivatives are calculated or the levels for the vertical derivatives. If
c     horizontal derivatives along model levels should be calculated, pass an
c     index array <vert(i,j,k)=k>.

      implicit none

c     Input and output parameters
      integer    nx,ny,nz,stride
      real       dfield(nx,ny,nz)
      real       field(nx,ny,nz)
      real       vert(nx,ny,nz)
      character  direction
      real       xmin,ymin,dx,dy
      real       mdv

c     Numerical and physical parameters
      real,parameter :: pi180=3.141592654/180.0
      real,parameter :: deltay=1.111775 ! transformation of units in 1/100km
      real,parameter :: zerodiv=0.00000001
      real,parameter :: eps=0.01

c     Auxiliary variables
      integer    i,j,k
      real       scale,lat
      real       vu,vl,vuvl,vlvu
      real       df(0:nx+1,0:ny+1,0:nz+1)
      real       f(0:nx+1,0:ny+1,0:nz+1)
      real       v(0:nx+1,0:ny+1,0:nz+1)

c     Check stride
      if (stride.le.0) then
         print*,'Invalid stride... Stop'
         stop
      elseif ( (stride.gt.1).and.
     >         ( ( (direction.ne.'x').and.
     >             (direction.ne.'y') ).or.
     >           (nz.gt.1) ) ) then
         print*,'Stride > 1 only implemented for 2D x/y... Stop'
         stop
      endif

c     Create the aura around the grid for fast boundary handling
      call aura (field,f,1,nx,ny,nz,xmin,ymin,dx,dy,mdv)
      call aura (vert, v,1,nx,ny,nz,xmin,ymin,dx,dy,mdv)

c     Vertical derivative
      if ((direction.eq.'z').or.
     >    (direction.eq.'th').or.
     >    (direction.eq.'p').or.
     >    (direction.eq.'m').and.
     >    (nz.gt.1)) then

c        Finite differencing
         do i=1,nx
            do j=1,ny
               do k=1,nz

                  if ((abs(f(i,j,k+1)-mdv).gt.eps).and.
     >                (abs(f(i,j,k-1)-mdv).gt.eps).and.
     >                (abs(v(i,j,k+1)-mdv).gt.eps).and.
     >                (abs(v(i,j,k-1)-mdv).gt.eps).and.
     >                (abs(v(i,j,  k)-mdv).gt.eps)) then

                     vu = v(i,j,  k)-v(i,j,k+1)
                     vl = v(i,j,k-1)-v(i,j,  k)
                     vuvl = vu/(vl+zerodiv)
                     vlvu = 1./(vuvl+zerodiv)

                     df(i,j,k) = 1./(vu+vl)
     >                           * (vuvl*(f(i,j,k-1)-f(i,j,  k))
     >                           +  vlvu*(f(i,j,  k)-f(i,j,k+1)))

                  else
                     df(i,j,k) = mdv
                  endif

               enddo
            enddo
         enddo

c     Horizontal derivative in the y direction: 3d
      elseif ((direction.eq.'y').and.(nz.gt.1)) then

c        Scale factor for derivative in 100km
         scale=1./(4.*dy*deltay)

c        Finite differencing
         do i=2,nx-1
            do j=2,ny-1
               do k=1,nz

                 if ((abs(f(i,j+2,k)-mdv).gt.eps).and.
     >               (abs(f(i,j-2,k)-mdv).gt.eps).and.
     >               (abs(f(i,j,k+1)-mdv).gt.eps).and.
     >               (abs(f(i,j,k-1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k+1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k+1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k-1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k-1)-mdv).gt.eps).and.
     >               (abs(v(i,j+2,k)-mdv).gt.eps).and.
     >               (abs(v(i,j+2,k)-mdv).gt.eps).and.
     >               (abs(v(i,j-2,k)-mdv).gt.eps).and.
     >               (abs(v(i,j-2,k)-mdv).gt.eps)) then

                     df(i,j,k) =
     >                   scale*(f(i,j+2,k)-f(i,j-2,k))
     >                  -(f(i,j,k+1)-f(i,j,k-1))/(v(i,j,k+1)-v(i,j,k-1))
     >                  *scale*(v(i,j+2,k)-v(i,j-2,k))

                  else
                     df(i,j,k) = mdv
                  endif

               enddo
            enddo
         enddo

c     Horizontal derivative in the x direction: 3d
      elseif ((direction.eq.'x').and.(nz.gt.1)) then

c        Finite differencing
         do j=2,ny-1

c           Scale factor for derivatives in 100km (latitude dependent)
            lat=ymin+real(j-1)*dy
            scale=1./(4.*dx*deltay*cos(pi180*lat)+zerodiv)

            do i=2,nx-1
               do k=1,nz

                 if ((abs(f(i+2,j,k)-mdv).gt.eps).and.
     >               (abs(f(i-2,j,k)-mdv).gt.eps).and.
     >               (abs(f(i,j,k+1)-mdv).gt.eps).and.
     >               (abs(f(i,j,k-1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k+1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k+1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k-1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k-1)-mdv).gt.eps).and.
     >               (abs(v(i+2,j,k)-mdv).gt.eps).and.
     >               (abs(v(i+2,j,k)-mdv).gt.eps).and.
     >               (abs(v(i-2,j,k)-mdv).gt.eps).and.
     >               (abs(v(i-2,j,k)-mdv).gt.eps)) then

                     df(i,j,k) =
     >                   scale*(f(i+2,j,k)-f(i-2,j,k))
     >                  -(f(i,j,k+1)-f(i,j,k-1))/(v(i,j,k+1)-v(i,j,k-1))
     >                  *scale*(v(i+2,j,k)-v(i-2,j,k))

                  else
                     df(i,j,k) = mdv
                  endif

               enddo
            enddo
         enddo

c     Horizontal derivative in the y direction: 2d
      elseif ((direction.eq.'y').and.(nz.eq.1)) then

c        Scale factor for derivative in 100km
         scale=1./(2.*dy*deltay*real(stride))

c        Finite differencing
         do i=5,nx-stride
            do j=5,ny-stride

               if ((abs(f(i,j+stride,1)-mdv).gt.eps).and.
     >             (abs(f(i,j-stride,1)-mdv).gt.eps)) then

                  df(i,j,1) = scale*(f(i,j+stride,1)-f(i,j-stride,1))

               else
                  df(i,j,1) = mdv
               endif

            enddo
         enddo

c     Horizontal derivative in the x direction: 2d
      elseif ((direction.eq.'x').and.(nz.eq.1)) then

c        Finite differencing
         do j=5,ny-stride

c           Scale factor for derivatives in 100km (latitude dependent)
            lat=ymin+real(j-1)*dy
            scale=1./(2.*dx*deltay*real(stride)*cos(pi180*lat)+zerodiv)

            do i=5,nx-stride

               if ((abs(f(i+stride,j,1)-mdv).gt.eps).and.
     >             (abs(f(i-stride,j,1)-mdv).gt.eps)) then

                  df(i,j,1) = scale*(f(i+stride,j,1)-f(i-stride,j,1))

               else
                  df(i,j,1) = mdv
               endif

            enddo
         enddo

c     Undefined direction for derivative
      else

         print*,'Invalid direction of derivative... Stop'
         stop

      endif

c     Get rid of the aura
      call aura (dfield,df,-1,nx,ny,nz,xmin,ymin,dx,dy,mdv)

      end


c     -----------------------------------------------------------------
c     Divergence
c     -----------------------------------------------------------------

      subroutine div(divgce,comp1,comp2,
     >               vert,xmin,ymin,dx,dy,nx,ny,nz,stride,mdv)

c     Calculate the divergence <divgce> of a vector field with the
c     components <comp1> and <comp2>.  The vertical coordinate is
c     specified in <vert>, the grid in <xmin,ymin,dx,dy,nx,ny,nz,mdv>.

      implicit none

c     Declaration of subroutine parameters
      integer nx,ny,nz,stride
      real    divgce(nx,ny,nz)
      real    comp1(nx,ny,nz),comp2(nx,ny,nz)
      real    tmp(nx,ny,nz)
      real    vert(nx,ny,nz)
      real    xmin,ymin,dx,dy
      real    mdv

c     Calculate the derivatives in x and y direction
      call deriv (tmp,comp1,'x',vert,xmin,ymin,dx,dy,
     >            nx,ny,nz,stride,mdv)
      call deriv (divgce,comp2,'y',vert,xmin,ymin,dx,dy,
     >            nx,ny,nz,stride,mdv)

      divgce=divgce+tmp

      end


c     -----------------------------------------------------------------
c     Computation of absolute value of 2D cartesian vector field
c     -----------------------------------------------------------------

      subroutine abs2d (xfield,yfield,absfield,nx,ny,nz,mdv)

      implicit none

      integer,intent(in) :: nx,ny,nz
      real,intent(in) :: mdv
      real,dimension(nx,ny,nz),intent(in) :: xfield,yfield
      real,dimension(nx,ny,nz),intent(out) :: absfield

      real,parameter :: eps=1.e-5

      absfield=sqrt(xfield**2+yfield**2)

      where ((abs(xfield-mdv).lt.eps).or.(abs(yfield-mdv).lt.eps))
     >  absfield=mdv

      end


c     -----------------------------------------------------------------
c     Computation of scalar product of 2d cartesian vector field
c     -----------------------------------------------------------------

      subroutine scal2d (xfield1,yfield1,xfield2,yfield2,
     >                   scalfield,nx,ny,nz,mdv)

      implicit none

      integer,intent(in) :: nx,ny,nz
      real,intent(in) :: mdv
      real,dimension(nx,ny,nz),intent(in) :: xfield1,yfield1
      real,dimension(nx,ny,nz),intent(in) :: xfield2,yfield2
      real,dimension(nx,ny,nz),intent(out) :: scalfield

      real,parameter :: eps=1.e-5

      scalfield=xfield1*xfield2+yfield1*yfield2

      where ((abs(xfield1-mdv).lt.eps).or.(abs(xfield2-mdv).lt.eps).or.
     >       (abs(yfield1-mdv).lt.eps).or.(abs(yfield2-mdv).lt.eps))
     >  scalfield=mdv

      end


c     -----------------------------------------------------------------
c     Standardization of 2d cartesian vector field
c     -----------------------------------------------------------------

      subroutine norm2d (xfield,yfield,nx,ny,nz,mdv)

c     Computation of unit 2d vectors / division by the absolute value

      implicit none

      integer,intent(in) :: nx,ny,nz
      real,intent(in) :: mdv
      real,dimension(nx,ny,nz),intent(inout) :: xfield,yfield

      real,dimension(nx,ny,nz) :: absfield

      real,parameter :: eps=1.e-6

      call abs2d(xfield,yfield,absfield,nx,ny,nz,mdv)

      where ((abs(xfield-mdv).gt.eps).and.(abs(yfield-mdv).gt.eps))
        xfield=xfield/absfield
        yfield=yfield/absfield
      endwhere

      end

c     -----------------------------------------------------------------
c     Calculate wind direction in degrees
c     -----------------------------------------------------------------

      subroutine winddir(nx,ny,nz,mdv,u3d,v3d,dir3d)

      ! calculation of wind direction of u and v field

      integer,intent(in) :: nx,ny,nz

      real,intent(in) :: mdv

      real,dimension(nx,ny,nz),intent(in) :: u3d,v3d

      real,dimension(nx,ny,nz) :: help

      real,dimension(nx,ny,nz),intent(out) :: dir3d

      real :: pi,twopi

      integer :: i,j,k

      real,parameter :: eps=1.e-5

      pi=4.*atan(1.)
      twopi=8.*atan(1.)

      do k=1,nz
        do j=1,ny
          do i=1,nx
            help(i,j,k)=atan(u3d(i,j,k)/v3d(i,j,k))
          enddo
        enddo
      enddo

      where (v3d>0.) help=pi+help
      where ((u3d>0.).and.(v3d<0.)) help=twopi+help

      dir3d=nint(360.*help/twopi)

      where (dir3d==360) dir3d=0

      where ((v3d==0.).and.(u3d<0.)) dir3d=90
      where ((v3d==0.).and.(u3d>0.)) dir3d=270
      where ((u3d==0.).and.(v3d<0.)) dir3d=0
      where ((u3d==0.).and.(v3d>0.)) dir3d=180
      where ((u3d==0.).and.(v3d==0.)) dir3d=mdv

      where ((abs(u3d-mdv).lt.eps).or.(abs(v3d-mdv).lt.eps))
     >      dir3d=mdv

      end


c     -----------------------------------------------------------------
      SUBROUTINE T_PHUV2LLUV(PHU,PHV,LLU,LLV,PHX,PHY)
c     -----------------------------------------------------------------

c     declaration of input/output vars:
      REAL        PHU,PHV,LLU,LLV,PHX,PHY

c     declaration of internal vars:
      real        phip,lamp,phipol,cosphip,sinphip,coslamp

      real      g2r
      parameter(g2r=0.0174533)

      real      pollon,pollat
      common /pole/ pollon,pollat

c     convert to radian:
      phip=PHY*g2r
      lamp=PHX*g2r
      phipol=pollat*g2r

c     abbrevations in rotpol.icl:
      zcospol=Cos(phipol)
      zsinpol=Sin(phipol)

c     other abbrevations:
      cosphip=Cos(phip)
      sinphip=Sin(phip)
      coslamp=Cos(lamp)

c     calculate the lat/lon u component:

      LLU=8*Cos(ASin(Coslamp*zcospol*Cosphip +
     -     zsinpol*Sinphip))*
     -   (-2*PHV*Sin(lamp - phipol) - 2*PHV*Sin(lamp + phipol) -
     -     PHU*Sin(lamp - phipol - phip) -
     -     2*PHU*Sin(phipol - phip) -
     -     PHU*Sin(lamp + phipol - phip) + PHU*Sin(lamp -
     -     phipol + phip) -
     -     2*PHU*Sin(phipol + phip) + PHU*Sin(lamp + phipol + phip))/
     -  (-20 + 4*Cos(2*lamp) + 2*Cos(2*(lamp - phipol)) -
     -     4*Cos(2*phipol) +
     -    2*Cos(2*(lamp + phipol)) - 4*Cos(lamp - 2*phipol - 2*phip) +
     -    4*Cos(lamp + 2*phipol - 2*phip) + 2*Cos(2*(lamp - phip)) +
     -    Cos(2*(lamp - phipol - phip)) + 6*Cos(2*(phipol - phip)) +
     -    Cos(2*(lamp + phipol - phip)) - 4*Cos(2*phip) +
     -    2*Cos(2*(lamp + phip)) + Cos(2*(lamp - phipol + phip)) +
     -    6*Cos(2*(phipol + phip)) + Cos(2*(lamp + phipol + phip)) +
     -    4*Cos(lamp - 2*phipol + 2*phip) -
     -    4*Cos(lamp + 2*phipol + 2*phip))

c     calculate the lat/lon v component:

      LLV=(PHV*Cosphip*zsinpol +
     -     zcospol*(-(PHU*Sin(lamp)) -
     -     PHV*Coslamp*Sinphip))/
     -     Sqrt(1 - (Coslamp*zcospol*Cosphip +
     -     zsinpol*Sinphip)**2)

      RETURN
      END


c     -----------------------------------------------------------------
c     Apply a special connecting filter for extraction of lines
c     -----------------------------------------------------------------

      subroutine linesplus(field,nx,ny,nz,xmin,ymin,dx,dy,mdv)

      ! connecting lines in 1/0 field

      implicit none

      integer,intent(in) :: nx,ny,nz
      real,intent(in) :: xmin,ymin,dx,dy,mdv
      real,dimension(nx,ny,nz),intent(inout) :: field

      integer :: i,j,k
      real,dimension(0:nx+1,0:ny+1,0:nz+1) :: f
      real,parameter :: eps=1.e-6

      call aura(field,f,1,nx,ny,nz,xmin,ymin,dx,dy,mdv)

      do k=1,nz
        do j=1,ny
          do i=1,nx
            if (f(i,j,k).gt.eps) cycle
            if ((f(i-1,j,k).gt.eps).and.(f(i+1,j,k).gt.eps)) then
              field(i,j,k)=1
              cycle
            endif
            if ((f(i,j-1,k).gt.eps).and.(f(i,j+1,k).gt.eps)) then
              field(i,j,k)=1
              cycle
            endif
            if ((f(i+1,j+1,k).gt.eps).and.(f(i-1,j-1,k).gt.eps)) then
              field(i,j,k)=1
              cycle
            endif
            if ((f(i+1,j-1,k).gt.eps).and.(f(i-1,j+1,k).gt.eps)) then
              field(i,j,k)=1
              cycle
            endif
            if ((f(i+1,j+1,k).gt.eps).and.
     >          ((f(i,j-1,k).gt.eps).or.(f(i-1,j,k).gt.eps))) then
              field(i,j,k)=1
              cycle
            endif
            if ((f(i+1,j-1,k).gt.eps).and.
     >          ((f(i,j+1,k).gt.eps).or.(f(i-1,j,k).gt.eps))) then
              field(i,j,k)=1
              cycle
            endif
            if ((f(i-1,j+1,k).gt.eps).and.
     >          ((f(i,j-1,k).gt.eps).or.(f(i+1,j,k).gt.eps))) then
              field(i,j,k)=1
              cycle
            endif
            if ((f(i-1,j-1,k).gt.eps).and.
     >          ((f(i,j+1,k).gt.eps).or.(f(i+1,j,k).gt.eps))) then
              field(i,j,k)=1
            endif
          enddo
        enddo
      enddo


      end
c     -----------------------------------------------------------------
c     Horizontal filter
c     -----------------------------------------------------------------

      subroutine filt2d (a,af,nx,ny,fil,misdat,
     &                   iperx,ipery,ispol,inpol)

c     Apply a conservative diffusion operator onto the 2d field a,
c     with full missing data checking.
c
c     a     real   inp  array to be filtered, dimensioned (nx,ny)
c     af    real   out  filtered array, dimensioned (nx,ny), can be
c                       equivalenced with array a in the calling routine
c     f1    real        workarray, dimensioned (nx+1,ny)
c     f2    real        workarray, dimensioned (nx,ny+1)
c     fil   real   inp  filter-coeff., 0<afil<=1. Maximum filtering with afil=1
c                       corresponds to one application of the linear filter.
c     misdat real  inp  missing-data value, a(i,j)=misdat indicates that
c                       the corresponding value is not available. The
c                       misdat-checking can be switched off with with misdat=0.
c     iperx int    inp  periodic boundaries in the x-direction (1=yes,0=no)
c     ipery int    inp  periodic boundaries in the y-direction (1=yes,0=no)
c     inpol int    inp  northpole at j=ny  (1=yes,0=no)
c     ispol int    inp  southpole at j=1   (1=yes,0=no)
c
c     Christoph Schaer, 1993

c     argument declaration
      integer     nx,ny
      real        a(nx,ny),af(nx,ny),fil,misdat
      integer     iperx,ipery,inpol,ispol

c     local variable declaration
      integer     i,j,is
      real        fh
      real        f1(nx+1,ny),f2(nx,ny+1)

c     compute constant fh
      fh=0.125*fil

c     compute fluxes in x-direction
      if (misdat.eq.0.) then
        do j=1,ny
        do i=2,nx
          f1(i,j)=a(i-1,j)-a(i,j)
        enddo
        enddo
      else
        do j=1,ny
        do i=2,nx
          if ((a(i,j).eq.misdat).or.(a(i-1,j).eq.misdat)) then
            f1(i,j)=0.
          else
            f1(i,j)=a(i-1,j)-a(i,j)
          endif
        enddo
        enddo
      endif
      if (iperx.eq.1) then
c       do periodic boundaries in the x-direction
        do j=1,ny
          f1(1,j)=f1(nx,j)
          f1(nx+1,j)=f1(2,j)
        enddo
      else
c       set boundary-fluxes to zero
        do j=1,ny
          f1(1,j)=0.
          f1(nx+1,j)=0.
        enddo
      endif

c     compute fluxes in y-direction
      if (misdat.eq.0.) then
        do j=2,ny
        do i=1,nx
          f2(i,j)=a(i,j-1)-a(i,j)
        enddo
        enddo
      else
        do j=2,ny
        do i=1,nx
          if ((a(i,j).eq.misdat).or.(a(i,j-1).eq.misdat)) then
            f2(i,j)=0.
          else
            f2(i,j)=a(i,j-1)-a(i,j)
          endif
        enddo
        enddo
      endif
c     set boundary-fluxes to zero
      do i=1,nx
        f2(i,1)=0.
        f2(i,ny+1)=0.
      enddo
      if (ipery.eq.1) then
c       do periodic boundaries in the x-direction
        do i=1,nx
          f2(i,1)=f2(i,ny)
          f2(i,ny+1)=f2(i,2)
        enddo
      endif
      if (iperx.eq.1) then
        if (ispol.eq.1) then
c         do south-pole
          is=(nx-1)/2
          do i=1,nx
            f2(i,1)=-f2(mod(i-1+is,nx)+1,2)
          enddo
        endif
        if (inpol.eq.1) then
c         do north-pole
          is=(nx-1)/2
          do i=1,nx
            f2(i,ny+1)=-f2(mod(i-1+is,nx)+1,ny)
          enddo
        endif
      endif

c     compute flux-convergence -> filter
      if (misdat.eq.0.) then
        do j=1,ny
        do i=1,nx
            af(i,j)=a(i,j)+fh*(f1(i,j)-f1(i+1,j)+f2(i,j)-f2(i,j+1))
        enddo
        enddo
      else
        do j=1,ny
        do i=1,nx
          if (a(i,j).eq.misdat) then
            af(i,j)=misdat
          else
            af(i,j)=a(i,j)+fh*(f1(i,j)-f1(i+1,j)+f2(i,j)-f2(i,j+1))
          endif
        enddo
        enddo
      endif
      end


c     --------------------------------------------------------------
c     Calculate potential temperature
c     --------------------------------------------------------------

      subroutine pottemp(pt,t,p,ie,je)

c     argument declaration
      integer   ie,je
      real      pt(ie,je),t(ie,je),p

c     variable declaration
      integer   i,j
      real      rdcp,tzero
      data      rdcp,tzero /0.286,273.15/

c     computation of potential temperature
      do i=1,ie
      do j=1,je
          if (t(i,j).lt.100.) then
            pt(i,j)=(t(i,j)+tzero)*( (1000./p)**rdcp )
          else
            pt(i,j)=t(i,j)*( (1000./p)**rdcp )
          endif
      enddo
      enddo
      end

c     --------------------------------------------------------------
c     Calculate quivalent potential temperature
c     --------------------------------------------------------------

      subroutine equpot(ap,t,q,p,ie,je)

c     calculate equivalent potential temperature

c     argument declaration
      integer  ie,je
      real     ap(ie,je),t(ie,je),q(ie,je),p

c     variable declaration
      integer  i,j
      real     rdcp,tzero
      data     rdcp,tzero /0.286,273.15/

c     computation of potential temperature
      do i=1,ie
         do j=1,je
               ap(i,j) = (t(i,j)+tzero)*(1000./p)
     +              **(0.2854*(1.0-0.28*q(i,j)))*exp(
     +              (3.376/(2840.0/(3.5*alog(t(i,j)+tzero)-alog(
     +              100.*p*max(1.0E-10,q(i,j))/(0.622+0.378*
     +              q(i,j)))-0.1998)+55.0)-0.00254)*1.0E3*
     +              max(1.0E-10,q(i,j))*(1.0+0.81*q(i,j)))
         enddo
      enddo
      end

c     -----------------------------------------------------------------
c     Shrink step
c     -----------------------------------------------------------------

      subroutine shrink_step (in_flag,nx,ny)

c     Take the input 0/1 array <in_flag(nx,ny)> and "erode" one point at
c     its boundary. The output is written to <out_flag(nx,ny)>. T

      implicit none

c     Declaration of input parameters
      integer nx,ny
      real    in_flag(nx,ny)

c     Auxiliary variables
      integer i,j
      integer il,ir,ju,jd
      real    out_flag(nx,ny)

c     Copy input to output
      do i=1,nx
         do j=1,ny
               out_flag(i,j)=in_flag(i,j)
         enddo
      enddo

c     Sweep through input and erode from output
      do i=1,nx
         do j=1,ny

c              Get neighbouring points
               ir=i+1
               if (ir.gt.nx) ir=nx
               il=i-1
               if (il.lt.1) il=1
               ju=j+1
               if (ju.gt.ny) ju=ny
               jd=j-1
               if (jd.lt.1) jd=1

c              Erode around boundary points
               if (in_flag(i,j).lt.0.5) then
                  out_flag(ir,j)=0.
                  out_flag(il,j)=0.
                  out_flag(ir,jd)=0.
                  out_flag(il,jd)=0.
                  out_flag(ir,ju)=0.
                  out_flag(il,ju)=0.
                  out_flag(i,jd)=0.
                  out_flag(i,ju)=0.
               endif

         enddo
      enddo

c     Copy output to input
      do i=1,nx
         do j=1,ny
               in_flag(i,j)=out_flag(i,j)
         enddo
      enddo

      return
      end

c     -----------------------------------------------
c     Grow step
c     -----------------------------------------------

      subroutine grow_step (in_flag,nx,ny)

c     Take the input 0/1 array <in_flag(nx,ny)> and "add" one point at
c     its boundary. The output is written to <out_flag(nx,ny)>.

      implicit none

c     Declaration of input parameters
      integer nx,ny
      real    in_flag(nx,ny)

c     Auxiliary variables
      integer i,j
      integer il,ir,ju,jd
      real    out_flag(nx,ny)

c     Copy input to output
      do i=1,nx
         do j=1,ny
              out_flag(i,j)=in_flag(i,j)
         enddo
      enddo

c     Sweep through input and erode from output
      do i=1,nx
         do j=1,ny

c              Get neighbouring points
               ir=i+1
               if (ir.gt.nx) ir=nx
               il=i-1
               if (il.lt.1) il=1
               ju=j+1
               if (ju.gt.ny) ju=ny
               jd=j-1
               if (jd.lt.1) jd=1

c              Erode around boundary points
               if (in_flag(i,j).gt.0.5) then

                  out_flag(ir,j)=1.
                  out_flag(il,j)=1.
                  out_flag(ir,jd)=1.
                  out_flag(il,jd)=1.
                  out_flag(ir,ju)=1.
                  out_flag(il,ju)=1.
                  out_flag(i,jd)=1.
                  out_flag(i,ju)=1.
               endif

         enddo
      enddo

c     Copy input to output
      do i=1,nx
         do j=1,ny
              in_flag(i,j)=out_flag(i,j)
         enddo
      enddo


      return
      end

