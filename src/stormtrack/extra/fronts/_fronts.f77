c     ---------------------------------------------------------------
      program fronts
c     ---------------------------------------------------------------

c     Atmospheric Dynamics Group IAC, ETH Zurich

c     Computation of location of fronts on THE charts
c     Advanced algorithm for front locating function used
c     Output with frontal strength

c     For advection speed: Output with negative sign for warm fronts 
c     and information of velocity and direction of motion

c     Optional output for frontal areas and removal of small objects
c     Optional removal of quasi-stationary fronts by min advection speeds

c     Literature:
c     -----------
c     Code is partly based on ideas of:
c     Hewson TD. 1998. Objective fronts. Met. Apps 5: 37–65, doi:10.1017/ S1350482798000553.

c     First development and application of our tool to COSMO Reanalysis:   
c     Jenkner J. et al. 2010. Detection and climatology of fronts in a high-resolution model reanalysis over the alps. Met. Apps 17: 1–18, doi:10.1002/met.142.

c     Introduction of frontal areas, mobile fronts, and first global climatology by our tool:
c     Schemm S. et al. 2014. Extratropical Fronts in the Lower Troposphere – Global Perspectives obtained from two Automated Methods. Quart. J. Roy. Meteor. Soc. 

c     History:
c     ---------
c     Johannes Jenkner April 2006
c                      September 2006
c                      September 2008
c     Sebastian Schemm October 2013
c     		           June 2014	

c     Development of front2line tool
c     Michael Sprenger April 2014

      use netcdf

      implicit none

c     ---------------------------------------------------------------
c     Some parameters
c     ---------------------------------------------------------------

      character*80 outformat
      parameter   (outformat='compress')
c      parameter   (outformat='2d')

c     ---------------------------------------------------------------
c     Declaration of variables
c     ---------------------------------------------------------------

      character(len=100) :: innam,outnam
      character(len=80)  :: invar,cfn

      integer :: iargc,ierr,grid
      integer :: idin,idout,idcst
      integer :: ndim,ntimes,nmin
      integer :: i,j,k,n,nz,nlev,ntot

      integer, dimension(4) :: vardim
      integer, dimension(5) :: stdate

      real :: thld
      real :: misdat
      real :: pollon,pollat
      real :: dx,dy
      real :: phx,phy
      real :: minadv

      real,dimension(    4) :: varmin,varmax,stag
      real,dimension(10000) :: time

      real,dimension(:),allocatable :: aklev,bklev,aklay,bklay

      real,dimension(:,:),allocatable :: clusterin

      real,dimension(:,:),allocatable :: gradx,grady,u2,v2,
     >                                   condi1,t2,qv2,tfield,farea,
     >                                   tfilt,fu,fv,vert,fvel

c     MSP
      integer,dimension(:,:),allocatable :: belowc,abovec
      real,dimension(:),allocatable      :: out1
      integer                               nbelow,nabove
      integer                               belowl(20000)
      integer                               il,ir,ju,jd
      integer                               ind,ival
      integer ierror,nx,ny
      real    xmin,ymin,xmax,ymax 
      real    lon1(2000),lat1(2000),pressure
      integer dimid,varid,cdfid,dimid1,dimid2
      character*80 varname,cdfname
      integer      nfilt,nfill
      integer      clust_count(1000000)
      integer      ilist,nlist
      integer      ierr
      integer      varid,dimid

      integer,dimension(:,:),allocatable :: clusterout,onecluster

      real,parameter :: eps=1.e-6

      logical :: winds,plus,withgap,area,mobile,objects,diffusive
     
c      common /pole/ pollon,pollat     

c     -----------------------------------------------
c     Check number of input arguments
c     -----------------------------------------------

      print*,"========================================================"
      print*," FRONT DETECTION "
      print*,"========================================================"
      print*
      
      if ( (iargc().lt.4).or.(iargc().gt.13) ) then
        write(*,*) 'USAGE: ./fronts [-winds]            ',
     >                                  ': with winds'
        write(*,*) '                [-plus]             ',
     >                                  ': broaden line'
        write(*,*) '                [-withgap]          ',
     >                                  ': fill gaps in mobile fronts'
        write(*,*) '                [-area nfill]       ',
     >                                  ': gap filling for frontal area'
        write(*,*) '                [-mobile thld]      ',
     >                                  ': threshold for mobile fronts'
        write(*,*) '                [-diffuisve nfilt]  ',
     >                                  ': # diffusive filtering steps'
        write(*,*) '                [-objects minpixel] ',
     >                                  ': remove small objects'
        write(*,*) '                INFILE'
        write(*,*) '                OUTFILE'
        write(*,*) '                VARIABLE'
        write(*,*) '                THRESHOLD '
	    print*, " "
	    stop
      end if

c     -----------------------------------------------------------------
c     Read input parameters and input file
c     -----------------------------------------------------------------

      i=1
      call getarg(i,invar)
      winds=(invar=='-winds')
      if (winds) i=i+1

      call getarg(i,invar) 
      plus=(invar=='-plus')
      if (plus) i=i+1

      call getarg(i,invar)
      withgap=(invar=='-withgap')
      if (withgap) i=i+1

      call getarg(i,invar)
      area=(invar=='-area')
      if (area) then
       i=i+1
        call getarg(i,invar)
        read(invar,*) nfill
        i=i+1
      endif

      call getarg(i,invar)
      mobile=(invar=='-mobile')
      if (mobile) then
       i=i+1	
        call getarg(i,invar)
        read(invar,*) minadv
        i=i+1
      endif

      call getarg(i,invar)
      diffusive=(invar=='-diffusive')
      if (diffusive) then
       i=i+1	
        call getarg(i,invar)
        read(invar,*) nfilt
        i=i+1
      endif

      call getarg(i,invar)
      objects=(invar=='-objects')
      if (objects) then
       i=i+1	
        call getarg(i,invar)
        read(invar,*) nmin
        i=i+1
      endif
      
      call getarg(i,innam)
      call getarg(i+1,outnam)
      call getarg(i+3,invar)
      read(invar,*) thld
      call getarg(i+2,invar)

c     The mobility filter only makes sense if winds are included
      if (mobile) then
        winds = .true.
      endif

c     Print Info
      print*
      print*,"----- PARAMETER ----------------------------------------"
      print*

      if (winds)     
     >         write(*,"(A)")      "-winds"
      if (plus)      
     >         write(*,"(A)")      "-plus"
      if (withgap)   
     >         write(*,"(A)")      "-withgap"
      if (area)      
     >         write(*,"(A)")      "-area",nfill
      if (mobile)    
     >         write(*,"(A,F4.1)") "-mobile :",minadv
      if (objects)   
     >         write(*,"(A,I4.1)") "-objects :",nmin
      if (diffusive) 
     >         write(*,"(A,I4.1)") "-diffusive :",nfilt

c     ------------------------------------------------------------------
c     Set grid parameters
c     ------------------------------------------------------------------

c     Open file
      ierror = NF90_OPEN(TRIM(innam),nf90_nowrite,cdfid)
      IF ( ierror /= nf90_NoErr ) PRINT *,NF90_STRERROR(ierror)

c     Get dimesnion in x direction
      ierror = nf90_inq_dimid(cdfid,'rlon',dimid)
      IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror)
      ierror = nf90_inquire_dimension(cdfid, dimid, len = nx)
      IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror)
      
c     Get dimesnion in y direction
      ierror = nf90_inq_dimid(cdfid,'rlat',dimid)
      IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror)
      ierror = nf90_inquire_dimension(cdfid, dimid, len = ny)
      IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror)

c     Get min and max longitude
      ierror = nf90_inq_varid(cdfid,'rlon', varid)
      IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror)
      ierror = NF90_GET_VAR(cdfid,varid,lon1(1:nx))
      IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror) 
      xmin = lon1( 1)
      xmax = lon1(nx)
      
c     Get min and max latitude
      ierror = nf90_inq_varid(cdfid,'rlat', varid)
      IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror)
      ierror = NF90_GET_VAR(cdfid,varid,lat1(1:ny))
      IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror) 
      ymin = lat1( 1)
      ymax = lat1(ny)

c     Get the grid resolution
      dx = ( lon1(nx) - lon1(1) ) / ( real(nx) - 1.)
      dy = ( lat1(ny) - lat1(1) ) / ( real(ny) - 1.)
      
c     Get pressure
      ierror = nf90_inq_varid(cdfid,'pressure', varid)
      IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror)
      ierror = NF90_GET_VAR(cdfid,varid,pressure)
      IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror) 
      
c     Close file
      ierror = NF90_CLOSE(cdfid)
      IF( ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror)

c     Copy grid info
      vardim(1) = nx
      vardim(2) = ny
      vardim(3) = 1
      varmin(1) = xmin
      varmax(1) = xmax
      varmin(2) = ymin
      varmax(2) = ymax
      varmin(3) = 0.01 * pressure
      varmax(3) = 0.01 * pressure
      nz        = 1
      misdat    = -999.
       
c     Determine the x topology of the grid
c     2: periodic, cyclic point; 
c     1: periodic, not cyclic; 
c     0: not periodic (and therefore not closed)

      if (abs(xmax-xmin-360.).lt.eps) then
         grid=2
      elseif (abs(xmax-xmin-360.+dx).lt.eps) then
         grid=1
      else
         grid=0
      endif

c     Print grid info
      print*
      print*,"----- GRID ---------------------------------------------"
      print*
      print*,nx,ny 
      print*,xmin,xmax 
      print*,ymin,ymax
      print*,dx,dy
      print*,pressure

c     ------------------------------------------------------------------
c     Allocate required arrays
c     ------------------------------------------------------------------

c     Input meteorological fields
      allocate(t2   (nx,ny))       ! temperature
      allocate(qv2  (nx,ny))       ! specific humidity
      allocate(u2   (nx,ny))       ! wind component in X direction
      allocate(v2   (nx,ny))       ! wind component in Y direction

c     Derived meteorological fields
      allocate(tfield(nx,ny))    ! T | TH | THE for front identification
      allocate(tfilt (nx,ny))    ! filtered array <tfield>
      allocate(gradx(nx,ny))     ! X gradient of <tfilt>
      allocate(grady(nx,ny))     ! Y gradient of <tfilt>

c     Frontal features
      allocate(farea(nx,ny))     ! frontal area
      allocate(condi1(nx,ny))    ! Condition 1 in front identification
      allocate(fvel (nx,ny))     ! wind speed parallel to grad(THE)


c     Arrays for clustering
      allocate(clusterout(nx,ny))
      allocate(clusterin(nx,ny))
      allocate(vert(nx,ny))

c     Define vert array
      do i=1,nx
        do j=1,ny
           vert(i,j) = 1.
        enddo
      enddo

c     -----------------------------------------------
c     Read input data
c     -----------------------------------------------
     
      print*
      print*,"----- INPUT DATA ---------------------------------------"
      print*
      
c     Open file
      ierror = NF90_OPEN(TRIM(innam),nf90_nowrite,cdfid)
      IF ( ierror /= nf90_NoErr ) PRINT *,NF90_STRERROR(ierror)

c     Get pressure
      ierror = nf90_inq_varid(cdfid,'pressure', varid)
      IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror)
      ierror = NF90_GET_VAR(cdfid,varid,pressure)
      IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror)            
      pressure = 0.01 * pressure
      print*,'R P -> 0.01 * P [Pa -> hPa]'
 
c     Read T
      ierror = nf90_inq_varid(cdfid,'T', varid)
      IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror)
      ierror = NF90_GET_VAR(cdfid,varid,t2)
      IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror)
      print*,'R T -> T -273.15 K [deg C]' 
      t2 = t2 - 273.15
       
c     Read QV
      ierror = nf90_inq_varid(cdfid,'QV', varid)
      IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror)
      ierror = NF90_GET_VAR(cdfid,varid,qv2)
      IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror) 
      print*,'R QV [kg/kg]' 

c     Read U
      if ( winds ) then
        ierror = nf90_inq_varid(cdfid,'U', varid)
        IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror)
        ierror = NF90_GET_VAR(cdfid,varid,u2)
        IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror) 
        print*,'R U -> U(de-staggered) [m/s] ' 
        do i=1,nx-1
        	do j=1,ny
        	   u2(i,j) = 0.5 * ( u2(i,j) + u2(i+1,j) )
        	enddo
        enddo
      endif
      
c     Read V
      if ( winds ) then
        ierror = nf90_inq_varid(cdfid,'V', varid)
        IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror)
        ierror = NF90_GET_VAR(cdfid,varid,v2)
        IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror) 
        print*,'R V -> V(de-staggered) [m/s]' 
        do i=1,nx
        	do j=1,ny-1
        	   v2(i,j) = 0.5 * ( v2(i,j) + v2(i,j+1) )
        	enddo
        enddo
      endif
    
c     Close file
      ierror = NF90_CLOSE(cdfid)
      IF( ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierror)
        
c     Define the final input field and save it in tfield
      if ( invar.eq.'T' ) then
         tfield = t2
      elseif ( invar.eq.'TH' ) then
         call pottemp(tfield,t2,pressure,nx,ny)
         print*,'C T,P -> TH'
      elseif ( invar.eq.'THE' ) then
         call equpot(tfield,t2,qv2,pressure,nx,ny)
         print*,'C T,P,QV -> THE'
      else
         print*,'Fronts not defined for ',trim(invar)
         stop
      endif
        
c     On request apply a diffusive filter - save in tfilt
      tfilt = tfield
      print*,'F ',trim(invar),nfilt
      do i=1,nfilt   
          call filt2d (tfilt,tfilt,nx,ny,1.0,misdat,0,0,0,0)
      enddo
      
c     ------------------------------------------------------------------   
c     FRONTAL AREA + ADVECTION SPEED in direction of grad(TFP)
c     ------------------------------------------------------------------

      print*
      print*,"----- FRONTAL AREA --------------------------------------"
      print*
      
c	  --- Gradient of THE -> (gradx,grady)
      print*,'C [ 1] grad(',trim(invar),')'
      call grad(gradx,grady,tfilt,vert,xmin,ymin,dx,dy,nx,ny,nz,misdat)

c     --- Absolute value of gradient of THE |(gradx,grady)| -> condi1
      print*,'C [ 2] |grad(',trim(invar),')|'
      call abs2d(gradx,grady,condi1,nx,ny,nz,misdat)
 
c     --- Save frontal area
      print*,'C [ 3] |grad(',trim(invar),')|  > ',thld,' K/100km'
      where ((abs(condi1-misdat).gt.eps).and.(condi1.gt.thld))
          farea=1
      elsewhere
          farea=0
      endwhere

c     Normalized grad(THE)
      print*,'C [ 4] N = grad( THE ) / | grad( THE ) |'
      call norm2d(gradx,grady,nx,ny,nz,misdat)

c     Advection speed in direction of grad(THE)
      print*,'C [ 5] Advection in direction of N'
      call scal2d(u2,v2,gradx,grady,fvel,nx,ny,nz,misdat)

c     --------------------------------------------------------------  
c     Remove very small objects
c     --------------------------------------------------------------  

      print*
      print*,"----- OBJECT FILTER -------------------------------------"
      print*

      if (objects) then	

c         Clustering of all objects
          print*,'C [ 1] Cluster all objects'
          do i=1,nx
            do j=1,ny
               if ( farea(i,j).lt.eps) then
                  clusterin(i,j) = 0.
               else
                  clusterin(i,j) = 1.
               endif
            enddo
          enddo
     	  call clustering(clusterout,ntot,clusterin,nx,ny,grid)

c         Remove objects that are too small
          print*,'C [ 2] Remove small clusters',nmin
          do i=1,ntot
             clust_count(i) = 0
          enddo
          do i=1,nx
            do j=1,ny
                ind = clusterout(i,j)
                if ( ind.ne.0 ) clust_count(ind) = clust_count(ind) + 1
            enddo
          enddo
          do i=1,ntot
             if ( clust_count(i).lt.nmin ) then
                clust_count(i) = 0
             else
                clust_count(i) = i
             endif
          enddo
          do i=1,nx
            do j=1,ny
                ind = clusterout(i,j)
                if ( ind.ne.0 ) then
                   if ( clust_count(ind).eq.0 ) then 
                      clusterout(i,j)   = 0
                      farea     (i,j)   = 0.
                   endif
                endif
            enddo
          enddo
       
      endif

c     --------------------------------------------------------------
c     Define output fields
c     --------------------------------------------------------------

      print*
      print*,"----- DEFINE OUTPUT FIELDS ------------------------------"
      print*

c     Gap filling
      print*,'C [ 1] Cluster all gap regions'
      do i=1,nx
         do j=1,ny
             if ( farea(i,j).lt.eps) then
                clusterin(i,j) = 1.
             else
                clusterin(i,j) = 0.
             endif
         enddo
      enddo
      call clustering(clusterout,ntot,clusterin,nx,ny,grid)

c     Keep only clusters surrounded by farea
      do i=1,ntot
           clust_count(i) = 1
      enddo
      do i=1,nx
         do j=1,ny

            ind = clusterout(i,j)
            if ( ind.ne.0 ) then

c              Get neighbours
               il=i-1
               ir=i+1
               ju=j+1
               jd=j-1
               if (ju.gt.ny) ju = ny
               if (jd.lt.1 ) jd = 1
               if (ir.gt.nx) ir = nx
               if (il.lt.1 ) il = 1

c              Remove the gap cluzster if connected to boundary
               if (ir.eq.nx ) clust_count(ind) = 0
               if (il.eq.1  ) clust_count(ind) = 0
               if (ju.eq.ny ) clust_count(ind) = 0
               if (jd.eq.1  ) clust_count(ind) = 0

            endif

         enddo
      enddo
      do i=1,nx
         do j=1,ny
             ind = clusterout(i,j)
             if ( ind.ne.0 ) then
                if ( clust_count(ind).eq.1 ) then
                      farea     (i,j)   = 1.
                endif
             endif
          enddo
      enddo

c     2) Scale output with frontal intensity, set vacant pixels to misdat
      print*,'C [ 2] frontal area (flag) -> frontal area( grad )'
      do i=1,nx
        do j=1,ny
          if ( farea(i,j).gt.0.) then
              farea(i,j) = condi1(i,j)
          endif
        enddo
      enddo

c     3) Save wind components
      print*,'C [ 3] save advection speed'
      do i=1,nx
        do j=1,ny
          if ( farea(i,j).eq.0.) then
             fvel(i,j) = misdat
          endif
        enddo
      enddo


c     --------------------------------------------------------------  
c      Output 
c     --------------------------------------------------------------  

      print*
      print*,"----- WRITE OUPUT ---------------------------------------"
      print*
 
      if ( outformat.eq.'2d' ) then

        cdfname = trim(outnam)
        varname = 'finp_'//trim(invar)
        call writecdf2D_compress(cdfname,varname,tfield,0.,
     >                      dx,dy,xmin,ymin,nx,ny,1,1)
 
        cdfname = trim(outnam)
        varname = 'farea_'//trim(invar)
        call writecdf2D_compress(cdfname,varname,farea,0.,
     >                      dx,dy,xmin,ymin,nx,ny,0,1)
     
        cdfname = trim(outnam)
        varname = 'fvel_'//trim(invar)
        call writecdf2D_compress(cdfname,varname,fvel,0.,
     >                      dx,dy,xmin,ymin,nx,ny,0,1)

      else if ( outformat.eq.'compress' ) then

         nlist = 0
         do i=1,nx
            do j=1,ny
                if ( farea(i,j).ne.0.) then
                  nlist = nlist + 1
                endif
            enddo
         enddo
         allocate(out1(nlist))

         print*,trim(outnam)

         ierr = nf90_create(path=outnam,cmode=nf90_clobber,ncid=cdfid)
         IF(ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)

         ierr=nf90_def_dim(cdfid,'list', nlist, dimID )
         IF(ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)

         varname = 'lon'
         ierr = nf90_def_var(cdfid,varname,NF90_FLOAT,(/ dimID /),varID)
         IF(ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
         ierr = nf90_put_att(cdfid, varID, "standard_name","loongitude")
         IF(ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
         varname = 'lat'
         ierr = nf90_def_var(cdfid,varname,NF90_FLOAT,(/ dimID /),varID)
         IF(ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
         ierr = nf90_put_att(cdfid, varID, "standard_name","latitude")
         IF(ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
         varname = 'farea_'//trim(invar)
         ierr = nf90_def_var(cdfid,varname,NF90_FLOAT,(/ dimID /),varID)
         IF(ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
         ierr = nf90_put_att(cdfid, varID,
     >                          "standard_name","frontal area")
         IF(ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
         varname = 'fvel_'//trim(invar)
         ierr = nf90_def_var(cdfid,varname,NF90_FLOAT,(/ dimID /),varID)
         IF(ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
         ierr = nf90_put_att(cdfid, varID,
     >                          "standard_name","frontal advection")
         IF(ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)

         ierr = nf90_put_att(cdfid, NF90_GLOBAL, 'input', innam)
         IF(ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
         ierr = nf90_put_att(cdfid, NF90_GLOBAL, 'nfill', nfill)
         IF(ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
         ierr = nf90_put_att(cdfid, NF90_GLOBAL, 'minadv', minadv)
         IF(ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
         ierr = nf90_put_att(cdfid, NF90_GLOBAL, 'nfilt', nfilt)
         IF(ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
         ierr = nf90_put_att(cdfid, NF90_GLOBAL, 'nmin', nmin)
         IF(ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
         ierr = nf90_put_att(cdfid, NF90_GLOBAL, 'invar', invar)
         IF(ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
         ierr = nf90_put_att(cdfid, NF90_GLOBAL, 'thld', thld)
         IF(ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
         ierr = nf90_put_att(cdfid, NF90_GLOBAL, 'pressure', pressure)
         IF(ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)

         ierr = nf90_enddef(cdfid)
         if (ierr.gt.0) then
            print*, 'An error occurred while attempting to ',
     >              'finish definition mode.'
            stop
         endif

         varname = 'lon'
         ilist = 0
         do i=1,nx
            do j=1,ny
              if ( farea(i,j).ne.0.) then
                ilist       = ilist + 1
                out1(ilist) = xmin + real(i-1) * dx
              endif
            enddo
         enddo
         ierr = NF90_INQ_VARID(cdfid,varname,varid)
         IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
         ierr = nf90_put_var(cdfid,varID  ,out1)

         varname = 'lat'
         ilist = 0
         do i=1,nx
            do j=1,ny
              if ( farea(i,j).ne.0.) then
                ilist       = ilist + 1
                out1(ilist) = ymin + real(j-1) * dy
              endif
            enddo
         enddo
         ierr = NF90_INQ_VARID(cdfid,varname,varid)
         IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
         ierr = nf90_put_var(cdfid,varID  ,out1)

         varname = 'farea_'//trim(invar)
         ilist = 0
         do i=1,nx
            do j=1,ny
              if ( farea(i,j).ne.0.) then
                ilist       = ilist + 1
                out1(ilist) = farea(i,j)
              endif
            enddo
         enddo
         ierr = NF90_INQ_VARID(cdfid,varname,varid)
         IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
         ierr = nf90_put_var(cdfid,varID  ,out1)

         varname = 'fvel_'//trim(invar)
         ilist = 0
         do i=1,nx
            do j=1,ny
              if ( farea(i,j).ne.0.) then
                ilist       = ilist + 1
                out1(ilist) = fvel(i,j)
              endif
            enddo
         enddo
         ierr = NF90_INQ_VARID(cdfid,varname,varid)
         IF(ierror /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)
         ierr = nf90_put_var(cdfid,varID  ,out1)

         ierr = NF90_CLOSE(cdfid)
         IF( ierr /= nf90_NoErr) PRINT *,NF90_STRERROR(ierr)

      endif

      print*," "
      print*," DONE"
      print*," "

      end program fronts

c     -----------------------------------------------
c     Subroutines to write the netcdf output file
c     -----------------------------------------------

      subroutine writecdf2D(cdfname,varname,arr,time,
     >                      dx,dy,xmin,ymin,nx,ny,crefile,crevar)

c     Create and write to the netcdf file <cdfname>. The variable
c     with name <varname> and with time <time> is written. The data
c     are in the two-dimensional array <arr>. The list <dx,dy,xmin,
c     ymin,nx,ny> specifies the output grid. The flags <crefile> and
c     <crevar> determine whether the file and/or the variable should
c     be created

      IMPLICIT NONE

c     Declaration of input parameters
      character*80 cdfname,varname
      integer      nx,ny
      real         arr(nx,ny)
      real         dx,dy,xmin,ymin
      real         time
      integer      crefile,crevar

c     Further variables
      real         varmin(4),varmax(4),stag(4)
      integer      ierr,cdfid,ndim,vardim(4)
      character*80 cstname
      real         mdv
      integer      datar(14),stdate(5)
      integer      i

C     Definitions 
      varmin(1)=xmin
      varmin(2)=ymin
      varmin(3)=1050.
      varmax(1)=xmin+real(nx-1)*dx
      varmax(2)=ymin+real(ny-1)*dy
      varmax(3)=1050.
      cstname='no_constants_file'
      ndim=4
      vardim(1)=nx
      vardim(2)=ny
      vardim(3)=1
      stag(1)=0.
      stag(2)=0.
      stag(3)=0.
      mdv=-999.98999

C     Create the file
      if (crefile.eq.0) then
         call cdfwopn(cdfname,cdfid,ierr)
         if (ierr.ne.0) goto 906
      else if (crefile.eq.1) then
         call crecdf(cdfname,cdfid,
     >        varmin,varmax,ndim,cstname,ierr)
         if (ierr.ne.0) goto 903 

C        Write the constants file
         datar(1)=vardim(1)
         datar(2)=vardim(2)
         datar(3)=int(1000.*varmax(2))
         datar(4)=int(1000.*varmin(1))
         datar(5)=int(1000.*varmin(2))
         datar(6)=int(1000.*varmax(1))
         datar(7)=int(1000.*dx)
         datar(8)=int(1000.*dy)
         datar(9)=1
         datar(10)=1
         datar(11)=0            ! data version
         datar(12)=0            ! cstfile version
         datar(13)=0            ! longitude of pole
         datar(14)=90000        ! latitude of pole     
         do i=1,5
            stdate(i)=0
         enddo
c         call wricst(cstname,datar,0.,0.,0.,0.,stdate)
      endif

c     Write the data to the netcdf file, and close the file
      if (crevar.eq.1) then
         call putdef(cdfid,varname,ndim,mdv,
     >        vardim,varmin,varmax,stag,ierr)
         if (ierr.ne.0) goto 904
      endif
      call putdat(cdfid,varname,time,0,arr,ierr)
      if (ierr.ne.0) goto 905
      call clscdf(cdfid,ierr)

      return
       
c     Error handling
 903  print*,'*** Problem to create netcdf file ***'
      stop
 904  print*,'*** Problem to write definition ***'
      stop
 905  print*,'*** Problem to write data ***'
      stop
 906  print*,'*** Problem to open netcdf file ***'
      stop

      END

      subroutine writecdf2D_compress(cdfname,varname,arr,time,
     >                      dx,dy,xmin,ymin,nx,ny,crefile,crevar)

c     Create and write to the netcdf file <cdfname>. The variable
c     with name <varname> and with time <time> is written. The data
c     are in the two-dimensional array <arr>. The list <dx,dy,xmin,
c     ymin,nx,ny> specifies the output grid. The flags <crefile> and
c     <crevar> determine whether the file and/or the variable should
c     be created

      IMPLICIT NONE

c     Declaration of input parameters
      character*80 cdfname,varname
      integer      nx,ny
      real         arr(nx,ny)
      real         dx,dy,xmin,ymin
      real         time
      integer      crefile,crevar

c     Further variables
      real         varmin(4),varmax(4),stag(4)
      integer      ierr,cdfid,ndim,vardim(4)
      character*80 cstname
      real         mdv
      integer      datar(14),stdate(5)
      integer      i

C     Definitions
      varmin(1)=xmin
      varmin(2)=ymin
      varmin(3)=1050.
      varmax(1)=xmin+real(nx-1)*dx
      varmax(2)=ymin+real(ny-1)*dy
      varmax(3)=1050.
      cstname='no_constants_file'
      ndim=4
      vardim(1)=nx
      vardim(2)=ny
      vardim(3)=1
      stag(1)=0.
      stag(2)=0.
      stag(3)=0.
      mdv=-999.98999

C     Create the file
      if (crefile.eq.0) then
         call cdfwopn(cdfname,cdfid,ierr)
         if (ierr.ne.0) goto 906
      else if (crefile.eq.1) then
         call crecdf(cdfname,cdfid,
     >        varmin,varmax,ndim,cstname,ierr)
         if (ierr.ne.0) goto 903

C        Write the constants file
         datar(1)=vardim(1)
         datar(2)=vardim(2)
         datar(3)=int(1000.*varmax(2))
         datar(4)=int(1000.*varmin(1))
         datar(5)=int(1000.*varmin(2))
         datar(6)=int(1000.*varmax(1))
         datar(7)=int(1000.*dx)
         datar(8)=int(1000.*dy)
         datar(9)=1
         datar(10)=1
         datar(11)=0            ! data version
         datar(12)=0            ! cstfile version
         datar(13)=0            ! longitude of pole
         datar(14)=90000        ! latitude of pole
         do i=1,5
            stdate(i)=0
         enddo
c         call wricst(cstname,datar,0.,0.,0.,0.,stdate)
      endif

c     Write the data to the netcdf file, and close the file
      if (crevar.eq.1) then
         call putdef(cdfid,varname,ndim,mdv,
     >        vardim,varmin,varmax,stag,ierr)
         if (ierr.ne.0) goto 904
      endif
      call putdat(cdfid,varname,time,0,arr,ierr)
      if (ierr.ne.0) goto 905
      call clscdf(cdfid,ierr)

      return

c     Error handling
 903  print*,'*** Problem to create netcdf file ***'
      stop
 904  print*,'*** Problem to write definition ***'
      stop
 905  print*,'*** Problem to write data ***'
      stop
 906  print*,'*** Problem to open netcdf file ***'
      stop

      END

