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
      integer, intent(in)	:: nx,ny
      integer, intent(in)	:: grid
      integer, intent(inout) 	:: outar(nx,ny)
      integer, intent(inout)  	:: ntot
      real, intent(in) 		:: inar(nx,ny)
      
c     Auxiliary variables
      integer			:: i,j
      
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

c     Correct the output array (so far, the index is 1 unit to large)
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
      integer, intent(in)	:: nx,ny
      integer, intent(in)	:: grid
      integer, intent(in)	:: label
      integer, intent(in) 	:: i,j
      integer, intent(inout)	:: outar(nx,ny)

c     Auxiliary variables
      integer  :: il,ir,ju,jd,im,jm
      integer  :: k
      integer  :: stack
      integer  :: indx(100000),indy(100000)
      integer  :: workar(0:nx+1,0:ny+1)

c     Initialize working array
      workar=0
      workar(1:nx,1:ny)=outar

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

c       Mark all connected elements (build up the stack)
        if (workar(il,jm).eq.1) then
          workar(il,jm)=label
          stack=stack+1
          indx(stack)=il
          indy(stack)=jm
        endif
        if (workar(ir,jm).eq.1) then
          workar(ir,jm)=label
          stack=stack+1
          indx(stack)=ir
          indy(stack)=jm
        endif
        if (workar(im,ju).eq.1) then
          workar(im,ju)=label
          stack=stack+1
          indx(stack)=im
          indy(stack)=ju
        endif
        if (workar(im,jd).eq.1) then
          workar(im,jd)=label
          stack=stack+1
          indx(stack)=im
          indy(stack)=jd
        endif
        if (workar(il,jd).eq.1) then
          workar(il,jd)=label
          stack=stack+1
          indx(stack)=il
          indy(stack)=jd
        endif
        if (workar(ir,jd).eq.1) then
          workar(ir,jd)=label
          stack=stack+1
          indx(stack)=ir
          indy(stack)=jd
        endif
        if (workar(il,ju).eq.1) then
          workar(il,ju)=label
          stack=stack+1
          indx(stack)=il
          indy(stack)=ju
        endif
        if (workar(ir,ju).eq.1) then
          workar(ir,ju)=label
          stack=stack+1
          indx(stack)=ir
          indy(stack)=ju
        endif

        if (stack.eq.0) exit wstack

      enddo wstack

      outar=workar(1:nx,1:ny)

c     -----------------------------------------------
      end subroutine connect
c     -----------------------------------------------


c     -----------------------------------------------------------------
c     Gradient
c     -----------------------------------------------------------------

      subroutine grad(gradx,grady,scalar,
     >                vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

c     Calculate the two vector components <gradx> and <grady> of the 
c     scalar field <scalar>.  The vertical coordinate is specified in
c     <vert>, the grid in <xmin,ymin,dx,dy,nx,ny,nz,mdv>.

      implicit none

c     Declaration of subroutine parameters
      integer nx,ny,nz
      real    gradx(nx,ny,nz)
      real    grady(nx,ny,nz)
      real    scalar(nx,ny,nz)
      real    vert(nx,ny,nz)
      real    xmin,ymin,dx,dy
      real    mdv

c     Calculate the derivatives in x and y direction
      call deriv (gradx,scalar,'x',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
      call deriv (grady,scalar,'y',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

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
     >                  vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

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
      integer    nx,ny,nz
      real       dfield(nx,ny,nz)
      real       field(nx,ny,nz)
      real       vert(nx,ny,nz)
      character  direction
      real       xmin,ymin,dx,dy
      real       mdv
      
c     Numerical and physical parameters
      real       pi180
      parameter  (pi180=3.141592654/180.)
      real       deltay
      parameter  (deltay=1.111775) ! transformation of units in 1/100km
      real       zerodiv
      parameter  (zerodiv=0.00000001)
      real       eps
      parameter  (eps=0.01) 

c     Auxiliary variables
      integer    i,j,k
      real       vmin,vmax
      real       scale,lat
      real       vu,vl,vuvl,vlvu
      real       df(0:nx+1,0:ny+1,0:nz+1)
      real       f(0:nx+1,0:ny+1,0:nz+1)
      real       v(0:nx+1,0:ny+1,0:nz+1)

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
         scale=1./(2.*dy*deltay)
         
c        Finite differencing
         do i=1,nx
            do j=1,ny
               do k=1,nz

                 if ((abs(f(i,j+1,k)-mdv).gt.eps).and.
     >               (abs(f(i,j-1,k)-mdv).gt.eps).and.  
     >               (abs(f(i,j,k+1)-mdv).gt.eps).and.  
     >               (abs(f(i,j,k-1)-mdv).gt.eps).and.  
     >               (abs(v(i,j,k+1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k+1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k-1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k-1)-mdv).gt.eps).and.
     >               (abs(v(i,j+1,k)-mdv).gt.eps).and.
     >               (abs(v(i,j+1,k)-mdv).gt.eps).and.
     >               (abs(v(i,j-1,k)-mdv).gt.eps).and.
     >               (abs(v(i,j-1,k)-mdv).gt.eps)) then

                     df(i,j,k) = 
     >                   scale*(f(i,j+1,k)-f(i,j-1,k)) 
     >                  -(f(i,j,k+1)-f(i,j,k-1))/(v(i,j,k+1)-v(i,j,k-1)) 
     >                  *scale*(v(i,j+1,k)-v(i,j-1,k))

                  else
                     df(i,j,k) = mdv
                  endif

               enddo
            enddo
         enddo

c     Horizontal derivative in the x direction: 3d
      elseif ((direction.eq.'x').and.(nz.gt.1)) then

c        Finite differencing
         do j=1,ny
            
c           Scale factor for derivatives in 100km (latitude dependent)
            lat=ymin+real(j-1)*dy
            scale=1./(2.*dx*deltay*cos(pi180*lat)+zerodiv)

            do i=1,nx
               do k=1,nz

                 if ((abs(f(i+1,j,k)-mdv).gt.eps).and.
     >               (abs(f(i-1,j,k)-mdv).gt.eps).and.  
     >               (abs(f(i,j,k+1)-mdv).gt.eps).and.  
     >               (abs(f(i,j,k-1)-mdv).gt.eps).and.  
     >               (abs(v(i,j,k+1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k+1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k-1)-mdv).gt.eps).and.
     >               (abs(v(i,j,k-1)-mdv).gt.eps).and.
     >               (abs(v(i+1,j,k)-mdv).gt.eps).and.
     >               (abs(v(i+1,j,k)-mdv).gt.eps).and.
     >               (abs(v(i-1,j,k)-mdv).gt.eps).and.
     >               (abs(v(i-1,j,k)-mdv).gt.eps)) then
                           
                     df(i,j,k) = 
     >                   scale*(f(i+1,j,k)-f(i-1,j,k)) 
     >                  -(f(i,j,k+1)-f(i,j,k-1))/(v(i,j,k+1)-v(i,j,k-1))
     >                  *scale*(v(i+1,j,k)-v(i-1,j,k))               
                     
                  else
                     df(i,j,k) = mdv
                  endif

               enddo
            enddo
         enddo

c     Horizontal derivative in the y direction: 2d
      elseif ((direction.eq.'y').and.(nz.eq.1)) then
         
c        Scale factor for derivative in 100km
         scale=1./(2.*dy*deltay)
         
c        Finite differencing
         do i=1,nx
            do j=1,ny
               
               if ((abs(f(i,j+1,1)-mdv).gt.eps).and.
     >             (abs(f(i,j-1,1)-mdv).gt.eps)) then 
                  
                  df(i,j,1) = scale*(f(i,j+1,1)-f(i,j-1,1)) 
                  
               else
                  df(i,j,1) = mdv
               endif
               
            enddo
         enddo

c     Horizontal derivative in the x direction: 2d
      elseif ((direction.eq.'x').and.(nz.eq.1)) then

c        Finite differencing
         do j=1,ny
            
c           Scale factor for derivatives in 100km (latitude dependent)
            lat=ymin+real(j-1)*dy
            scale=1./(2.*dx*deltay*cos(pi180*lat)+zerodiv)

            do i=1,nx

               if ((abs(f(i+1,j,1)-mdv).gt.eps).and.
     >             (abs(f(i-1,j,1)-mdv).gt.eps)) then  
                           
                  df(i,j,1) = scale*(f(i+1,j,1)-f(i-1,j,1)) 
                  
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
     >               vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

c     Calculate the divergence <divgce> of a vector field with the 
c     components <comp1> and <comp2>.  The vertical coordinate is 
c     specified in <vert>, the grid in <xmin,ymin,dx,dy,nx,ny,nz,mdv>.

      implicit none

c     Declaration of subroutine parameters
      integer nx,ny,nz
      real    divgce(nx,ny,nz)
      real    comp1(nx,ny,nz),comp2(nx,ny,nz)
      real    tmp(nx,ny,nz)
      real    vert(nx,ny,nz)
      real    xmin,ymin,dx,dy
      real    mdv

c     Calculate the derivatives in x and y direction
      call deriv (tmp,comp1,'x',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)
      call deriv (divgce,comp2,'y',vert,xmin,ymin,dx,dy,nx,ny,nz,mdv)

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
c     Check environment of gridpoint for zero crossings
c     -----------------------------------------------------------------

      function zcross(f33,mdv) result(zout)

c     Check adjacent gridpoints for sign
c     Return true if zero crossing present and center gridpoint
c     closest to zero compared to the others

      implicit none

      real,dimension(3,3),intent(in) :: f33
      real,intent(in) :: mdv
      real,dimension(1,9) :: fcalc 
      real,parameter :: eps=1.e-6
      integer :: i
      logical :: zout

      fcalc=reshape(f33,(/ 1,9 /))

      zout=(abs(fcalc(5)-mdv).gt.eps)

      if (.not.zout) return

      where (abs(fcalc-mdv).lt.eps) fcalc=fcalc(5)

      do i=1,4
        zout=((sign(1.,fcalc(1,i)).ne.sign(1.,fcalc(1,5))).and.
     >      (abs(fcalc(1,5)).le.abs(fcalc(1,i))))        
        if (zout) return  
      enddo
      do i=6,9
        zout=((sign(1.,fcalc(1,i)).ne.sign(1.,fcalc(1,5))).and.
     >      (abs(fcalc(1,5)).le.abs(fcalc(1,i))))  
        if (zout) return
      enddo       

      end 


c     -----------------------------------------------------------------
c     Determine zero contours in gridded data
c     -----------------------------------------------------------------

      subroutine zcontour (field,nx,ny,nz,xmin,ymin,
     >                     dx,dy,mdv)

c     Locate zero contour lines in a gridded scalar field
c     input/output : field
c     (contains ones for contour line and zeros elsewhere)

      implicit none

      integer,intent(in) :: nx,ny,nz
      real,intent(in) :: xmin,ymin,dx,dy,mdv
      real,dimension(nx,ny,nz),intent(inout) :: field

      integer :: i,j,k
      real,dimension(0:nx+1,0:ny+1,0:nz+1) :: f      
      real,parameter :: eps=1.e-6

      logical,external :: zcross

      call aura(field,f,1,nx,ny,nz,xmin,ymin,dx,dy,mdv)      
      
      field=0.

      do k=1,nz
        do j=1,ny
          do i=1,nx
            if (zcross(f(i-1:i+1,j-1:j+1,k),mdv)) field(i,j,k)=1.
          enddo
        enddo
      enddo

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

      real :: pi,twopi,temp

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
