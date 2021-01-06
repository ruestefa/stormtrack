      PROGRAM front2line

c     ********************************************************************
c     * Read netCDF fronts and write them as lines                       *
c     * Michael Sprenger / Summer 2004                                   *
c     ********************************************************************

      IMPLICIT NONE

c     -------------------------------------------------------------------
c     Declaration of input parameters
c     -------------------------------------------------------------------

c     Input front file with fronts_THE, fronts_VEL, fronts_DIR
      character*80                        inp_fronts

c     Output netCDF file
      character*80                        out_filename

c     Minimal length of frontal lines (typically 500 km for ERAi)
      real                                inp_minlength

c     Smoothing parameter for smoothing cubic spline (typically 15 for ERAi)
      integer                             smoothing

c     Distance between vertices after reparameterization (in km)
      integer                             reparam

C     Gap briding distance (km) / Separation distance for frontal systems
      real                                inp_distance

c     -------------------------------------------------------------------
c     Internal parameters
c     -------------------------------------------------------------------

c     Test flag and file
      integer                             test
      parameter                           (test=0)
      character*80                        testfile
      parameter                           (testfile='TEST')

c     Numerical epsilon
      real                                eps
      parameter                           (eps=1.e-5)

c     Number of iteration for bulk elminiation, thinning, line combining
      integer                             niter
      parameter                           (niter=10)

c     Maximu size of working arrays (list of coordinates)
      integer                             maxlen
      parameter                           (maxlen=100000)

c     Maximum number of brancning steps in backtracking
      integer                             maxvar
      parameter                           (maxvar=10)

c     Number of extension steps for THE and FVEL extraction
      integer                             nexten
      parameter                           (nexten=2)

c     -------------------------------------------------------------------
c     Declaration of variables
c     -------------------------------------------------------------------

c     Grid parameters and 2d fields
      real                                    dx,dy
      real                                    xmin,ymin,xmax,ymax
      integer                                 nx,ny
      real,allocatable, dimension (:,:)    :: field_the
      real,allocatable, dimension (:,:)    :: field_fvel
      real,allocatable, dimension (:,:)    :: field_out1,field_out2
      integer,allocatable, dimension (:,:) :: field_inp
      integer,allocatable, dimension (:,:) :: field_system
      integer,allocatable, dimension (:,:) :: field_flag
      integer,allocatable, dimension (:,:) :: cluster
      integer                                 ncluster
      character*80                            fieldname

c     Frontal lines - state, length, sysid, indices, coordinates
      real                                 segm_length(maxlen)
      integer                              segm_sysid(maxlen)
      integer                              isegm0(maxlen),isegm1(maxlen)
      integer                              nsegm,segm_stat(maxlen)
      real                                 lilon(maxlen),lilat(maxlen)
      integer                              nline
      real                                 the(maxlen),fvel(maxlen)

c     netCDF variables
      integer                              cdfid
      integer                              nvars
      character*80                         vnam(100)
      real                                 varmin(4),varmax(4),stag(4)
      integer                              vardim(4)
      real                                 mdv
      integer                              ndim
      real                                 time(100)
      integer                              ntimes
      integer                              ierr,stat
      character*80                         varname,cdfname
      integer                              crefile

c     Variables for cubic spline interpolation
      real                                 x(maxlen),y(maxlen),w(maxlen)
      real                                 xb,xe,fp_lon,fp_lat,s
      integer                              nest,m,ier,iopt
      real                                 t_lon(maxlen),c_lon(maxlen)
      real                                 t_lat(maxlen),c_lat(maxlen)
      real                                 wrk(maxlen)
      integer                              n_lon,n_lat
      integer                              lwrk,iwrk(maxlen)
      real                                 y_lon(maxlen),y_lat(maxlen)

c     Segment and line searching - backtracking
      integer                              stackx(maxlen),stacky(maxlen)
      integer                              nstack
      integer                              contflag
      integer                              trlen
      real                                 trlon(maxlen),trlat(maxlen)
      integer                              stackp(maxlen)
      integer                              ntrack
      integer                              trackx(maxlen),tracky(maxlen)

c     Auxiliary variables
      integer                              isok
      integer                              i,j,k,l,n
      integer                              il,ir,ju,jd,i0,j0
      integer                              indx,indy
      character                            ch
      real                                 lon1,lat1,lon2,lat2,dist
      integer                              flag
      real                                 maxdist,mindist
      real                                 nstep
      integer                              ilabel
      integer                              dist0,dist1
      real                                 d1,d2,d3
      real                                 length
      integer                              icount
      real                                 rsum1,rsum2
      real                                 smooth

c     Externals
      real                                 sdis
      external                             sdis

c     -------------------------------------------------------------------
c     Preparations
c     -------------------------------------------------------------------

c     Read in parameter file
      open(10,file='fort.10')
       read(10,*) inp_fronts
       read(10,*) out_filename
       read(10,*) inp_minlength
       read(10,*) smoothing
       read(10,*) reparam
       read(10,*) inp_distance
       read(10,*) fieldname
      close(10)

      print*,'Fronts        : ',trim(inp_fronts)
      print*,'Out filename  : ',trim(out_filename)
      print*,'Front length  : ',inp_minlength
      print*,'Gap bridging  : ',inp_distance
      print*,'fieldname     : ',trim(fieldname)

c     Open netcDF file and constants file
      call cdfopn(inp_fronts,cdfid,ierr)
      if (ierr.ne.0) goto 998

c     Check variables on the file
      call getvars(cdfid,nvars,vnam,ierr)
      isok=0
      varname = 'fronts_'//trim(fieldname)
      call check_varok (isok,varname,vnam,nvars)
      varname = 'fvel_'//trim(fieldname)
      call check_varok (isok,varname,vnam,nvars)
      if (isok.ne.2) goto 998

c     Get grid description
      varname = 'fronts_'//trim(fieldname)
      call getdef(cdfid,varname,ndim,mdv,vardim,
     >            varmin,varmax,stag,ierr)
      if (ierr.ne.0) goto 998
      nx   = vardim(1)
      ny   = vardim(2)
      xmin = varmin(1)
      ymin = varmin(2)
      xmax = varmax(1)
      ymax = varmax(2)
      call gettimes(cdfid,time,ntimes,ierr)
      call clscdf(cdfid,ierr)
      dx = (xmax -xmin) / real(nx-1)
      dy = (ymax -ymin) / real(ny-1)

      print*,'xmin,xmax     : ',xmin,xmax
      print*,'ymin,ymax     : ',ymin,ymax
      print*,'dx,dy         : ',dx,dy
      print*,'time          : ',time(ntimes)

c     Allocate memory for input and output field
      allocate(field_the  (nx,ny),stat=stat)
      if (stat.ne.0) print*,'*** error allocating array field_the   ***'
      allocate(field_fvel (nx,ny),stat=stat)
      if (stat.ne.0) print*,'*** error allocating array field_fvel  ***'
      allocate(field_flag (nx,ny),stat=stat)
      if (stat.ne.0) print*,'*** error allocating array field_flag  ***'
      allocate(field_inp (nx,ny),stat=stat)
      if (stat.ne.0) print*,'*** error allocating array field_inp   ***'
      allocate(cluster   (nx,ny),stat=stat)
      if (stat.ne.0) print*,'*** error allocating array cluster     ***'
      allocate(field_system(nx,ny),stat=stat)
      if (stat.ne.0) print*,'*** error allocating array field_system***'
      allocate(field_out1(nx,ny),stat=stat)
      if (stat.ne.0) print*,'*** error allocating array field_out1  ***'
      allocate(field_out2(nx,ny),stat=stat)
      if (stat.ne.0) print*,'*** error allocating array field_out2  ***'

c     Read fronts
      call cdfopn(inp_fronts,cdfid,ierr)
      if (ierr.ne.0) goto 998

      varname = 'fronts_'//trim(fieldname)
      print*,'R ',trim(varname),' <- ',trim(inp_fronts)
      call getdat(cdfid,varname,time(ntimes),0,field_the,ierr)

      varname = 'fvel_'//trim(fieldname)
      print*,'R ',trim(varname),' <- ',trim(inp_fronts)
      call getdat(cdfid,varname,time(ntimes),0,field_fvel,ierr)

      call clscdf(cdfid,ierr)

c     Define frontal flag
      do i=1,nx
         do j=1,ny
         	if ( field_the(i,j).gt.0.) then
         		field_flag(i,j) = 1
            else
                field_flag(i,j) = 0
            endif
         enddo
      enddo

c     Save the input field
      do i=1,nx
        do j=1,ny
           field_inp(i,j) = field_flag(i,j)
        enddo
      enddo

c     Write flag field to test file
      if ( test.eq.1 ) then
        cdfname=testfile
        varname='INPUT'
        nstep  = 0.
        call writecdf2D(cdfname,varname,field_flag,nstep,
     >                  dx,dy,xmin,ymin,nx,ny,1,1)
      endif

c     -----------------------------------------------------------------
c     Remove bulky features
c     -----------------------------------------------------------------

      do k=1,niter

c     Mark interior points
      do i=1,nx
        do j=1,ny

            il = i-1
            if (il.lt.1) il=nx
            ir = i+1
            if ( ir.gt.nx ) ir=1
            jd = j-1
            if (jd.lt.1) jd=1
            ju = j+1
            if ( ju.gt.ny ) ju=ny

            if ( (field_flag(ir, j).ge.1).and.
     >           (field_flag(i , j).ge.1).and.
     >           (field_flag(il, j).ge.1).and.
     >           (field_flag(ir,jd).ge.1).and.
     >           (field_flag(i ,jd).ge.1).and.
     >           (field_flag(il,jd).ge.1).and.
     >           (field_flag(il,ju).ge.1).and.
     >           (field_flag(i ,ju).ge.1).and.
     >           (field_flag(ir,ju).ge.1) )
     >      then
                field_flag(i,j) = 2
            endif

         enddo
      enddo

c     Remove boundary points which also are connected to an interior point
      do i=1,nx
        do j=1,ny

            if ( field_flag(i,j).eq.1 ) then

                il = i-1
                if (il.lt.1) il=nx
                ir = i+1
                if ( ir.gt.nx ) ir=1
                jd = j-1
                if (jd.lt.1) jd=1
                ju = j+1
                if ( ju.gt.ny ) ju=ny

                if ( (field_flag(ir, j).eq.2).or.
     >               (field_flag(il, j).eq.2).or.
     >               (field_flag(ir,jd).eq.2).or.
     >               (field_flag(i ,jd).eq.2).or.
     >               (field_flag(il,jd).eq.2).or.
     >               (field_flag(ir,ju).eq.2).or.
     >               (field_flag(i ,ju).eq.2).or.
     >               (field_flag(il,ju).eq.2) )
     >          then
                  field_flag(i , j) = 0
                endif

           endif

         enddo
      enddo

c     Reset the status of each point
      do i=1,nx
        do j=1,ny
           if ( field_flag(i,j).ge.1 ) field_flag(i,j) = 1
        enddo
      enddo

      enddo

c     Write flag field to test file
      if ( test.eq.1 ) then
        cdfname=testfile
        varname='BULK'
        nstep  = 0.
        call writecdf2D(cdfname,varname,field_flag,nstep,
     >                  dx,dy,xmin,ymin,nx,ny,0,1)
      endif

c     -------------------------------------------------------------------
c     Thinning - see article thinning.pdf for details
c     Reduce the bulky lines to one-grid-point-connected lines
c     -------------------------------------------------------------------

      do n=1,niter

c     Reset counter for changes
      icount = 0


c     D1
      do i=1,nx
         do j=1,ny

            il = i-1
            if (il.lt.1) il=nx
            ir = i+1
            if ( ir.gt.nx ) ir=1
            jd = j-1
            if (jd.lt.1) jd=1
            ju = j+1
            if ( ju.gt.ny ) ju=ny

            if ( (field_flag( i, j).eq.1).and.
     >           (field_flag(ir, j).eq.1).and.
     >           (field_flag( i,jd).eq.1).and.
     >           (field_flag(il,ju).eq.0).and.
     >           (field_flag(i, ju).eq.0).and.
     >           (field_flag(il,j ).eq.0) )
     >      then
               field_flag(i,j) = 0
               icount = icount + 1
            endif
               
         enddo
      enddo

c     D2
      do i=1,nx
         do j=1,ny

            il = i-1
            if (il.lt.1) il=nx
            ir = i+1
            if ( ir.gt.nx ) ir=1
            jd = j-1
            if (jd.lt.1) jd=1
            ju = j+1
            if ( ju.gt.ny ) ju=ny

            if ( (field_flag( i, j).eq.1).and.
     >           (field_flag(il, j).eq.1).and.
     >           (field_flag( i,jd).eq.1).and.
     >           (field_flag( i,ju).eq.0).and.
     >           (field_flag(ir,ju).eq.0).and.
     >           (field_flag(ir,j ).eq.0) )
     >      then
               field_flag(i,j) = 0
               icount = icount + 1
            endif

         enddo
      enddo
                       
c     D3
      do i=1,nx
         do j=1,ny

            il = i-1
            if (il.lt.1) il=nx
            ir = i+1
            if ( ir.gt.nx ) ir=1
            jd = j-1
            if (jd.lt.1) jd=1
            ju = j+1
            if ( ju.gt.ny ) ju=ny

            if ( (field_flag( i, j).eq.1).and.
     >           (field_flag(il, j).eq.1).and.
     >           (field_flag( i,ju).eq.1).and.
     >           (field_flag(ir,j ).eq.0).and.
     >           (field_flag(i, jd).eq.0).and.
     >           (field_flag(ir,jd).eq.0) )
     >      then
               field_flag(i,j) = 0
               icount = icount + 1
            endif

         enddo
      enddo
         
c     D4
      do i=1,nx
         do j=1,ny

            il = i-1
            if (il.lt.1) il=nx
            ir = i+1
            if ( ir.gt.nx ) ir=1
            jd = j-1
            if (jd.lt.1) jd=1
            ju = j+1
            if ( ju.gt.ny ) ju=ny

            if ( (field_flag( i, j).eq.1).and.
     >           (field_flag(ir, j).eq.1).and.
     >           (field_flag( i,ju).eq.1).and.
     >           (field_flag(il,j ).eq.0).and.
     >           (field_flag(il,jd).eq.0).and.
     >           (field_flag(i, jd).eq.0 ) )
     >      then
               field_flag(i,j) = 0
               icount = icount + 1
            endif

         enddo
      enddo

c     E1
      do i=1,nx
         do j=1,ny

            il = i-1
            if (il.lt.1) il=nx
            ir = i+1
            if ( ir.gt.nx ) ir=1
            jd = j-1
            if (jd.lt.1) jd=1
            ju = j+1
            if ( ju.gt.ny ) ju=ny

            if ( (field_flag( i, j).eq.1).and.
     >           (field_flag(il, j).eq.1).and.
     >           (field_flag(ir,j ).eq.1).and.
     >           (field_flag( i,jd).eq.1).and.
     >           (field_flag( i,ju).eq.0) )
     >      then
               field_flag(i,j) = 0
               icount = icount + 1
            endif

         enddo
      enddo
      
c     E2
      do i=1,nx
         do j=1,ny

            il = i-1
            if (il.lt.1) il=nx
            ir = i+1
            if ( ir.gt.nx ) ir=1
            jd = j-1
            if (jd.lt.1) jd=1
            ju = j+1
            if ( ju.gt.ny ) ju=ny  

            if ( (field_flag( i, j).eq.1).and.
     >           (field_flag(il, j).eq.1).and.
     >           (field_flag( i,ju).eq.1).and.
     >           (field_flag( i,jd).eq.1).and.
     >           (field_flag(ir,j ).eq.0) )
     >      then
               field_flag(i,j) = 0
               icount = icount + 1
            endif
               
         enddo
      enddo
      
c     E3
      do i=1,nx
         do j=1,ny

            il = i-1
            if (il.lt.1) il=nx
            ir = i+1
            if ( ir.gt.nx ) ir=1
            jd = j-1
            if (jd.lt.1) jd=1
            ju = j+1
            if ( ju.gt.ny ) ju=ny  

            if ( (field_flag( i, j).eq.1).and.
     >           (field_flag(il, j).eq.1).and.
     >           (field_flag(ir, j).eq.1).and.
     >           (field_flag( i,ju).eq.1).and.
     >           (field_flag(i, jd).eq.0) )
     >      then
               field_flag(i,j) = 0
               icount = icount + 1
            endif

         enddo
      enddo

c     E4
      do i=1,nx
         do j=1,ny

            il = i-1
            if (il.lt.1) il=nx
            ir = i+1
            if ( ir.gt.nx ) ir=1
            jd = j-1
            if (jd.lt.1) jd=1
            ju = j+1
            if ( ju.gt.ny ) ju=ny  
            
            if ( (field_flag( i, j).eq.1).and.
     >           (field_flag(ir, j).eq.1).and.
     >           (field_flag( i,ju).eq.1).and.
     >           (field_flag( i,jd).eq.1).and.
     >           (field_flag(il,j ).eq.0) )
     >      then 
               field_flag(i,j) = 0
               icount = icount + 1
            endif

         enddo
      enddo

      enddo

c     Write flag field to test file
      if ( test.eq.1 ) then
        cdfname=testfile
        varname='THINNING'
        nstep  = 0.
        call writecdf2D(cdfname,varname,field_flag,nstep,
     >                  dx,dy,xmin,ymin,nx,ny,0,1)
      endif

c     -------------------------------------------------------------------
c     Determine line segments; the starting point can be still in the
c     middle of a line - different line segments will later be connected
c     The line segmensts are not yet ordered according to their length
c     -------------------------------------------------------------------

c     Do a clustering of the objects based on grid connectivity; the
c     clusters are needed to restrict gap bridging only between different
c     clusters
      call clustering(cluster,ncluster,field_flag,nx,ny)

c     Write flag field to test file
      if ( test.eq.1 ) then
        cdfname=testfile
        varname='CLUSTER'
        nstep  = 0.
        call writecdf2D(cdfname,varname,cluster,nstep,
     >                  dx,dy,xmin,ymin,nx,ny,0,1)
      endif

c     Init index of line segment array; nline counts the number of line
c     segments attributed to a frontal system
      nline = 0
      nstep = 0

c     Loop over all possible frontal systems - many line segements
 100  continue

c     Find a starting point for a frontal system: First try to find an
c     ending point with only one connecting neighbour
      indx = 0
      indy = 0
      do i=1,nx
        do j=1,ny
            if ( field_flag(i,j).ne.0) then

c             Get neighbours of potential starting point
              i0 = indx
              j0 = indy
              il = indx-1
              if (il.lt.1) il=nx
              ir = indx+1
              if ( ir.gt.nx ) ir=1
              jd = indy-1
              if (jd.lt.1) jd=1
              ju = indy+1
              if ( ju.gt.ny ) ju=ny

c             Count number of neighbours
              icount = 0
              if ( field_flag(ir, j).ne.0) icount = icount+1
              if ( field_flag(il, j).ne.0) icount = icount+1
              if ( field_flag(ir,jd).ne.0) icount = icount+1
              if ( field_flag(i ,jd).ne.0) icount = icount+1
              if ( field_flag(il,jd).ne.0) icount = icount+1
              if ( field_flag(ir,ju).ne.0) icount = icount+1
              if ( field_flag(i ,ju).ne.0) icount = icount+1
              if ( field_flag(il,ju).ne.0) icount = icount+1

c             Accept it if only one neighbour found
              if ( icount.eq.1 ) then
                 indx = i
                 indy = j
                 goto 110
              endif

            endif
        enddo
      enddo

c     Find a starting point for a frontal system: Try to find a candidate
c     witzh two neighbouring points, and the neighbours are itself
c     connected
      indx = 0
      indy = 0
      do i=1,nx
        do j=1,ny
            if ( field_flag(i,j).ne.0) then

c             Get neighbours of potential starting point
              i0 = indx
              j0 = indy
              il = indx-1
              if (il.lt.1) il=nx
              ir = indx+1
              if ( ir.gt.nx ) ir=1
              jd = indy-1
              if (jd.lt.1) jd=1
              ju = indy+1
              if ( ju.gt.ny ) ju=ny

c             Count number of neighbours
              icount = 0
              if ( ( field_flag(il, ju).ne.0 ).and.
     >             ( field_flag(i , ju).ne.0 ) ) icount = icount +1
              if ( ( field_flag(i , ju).ne.0 ).and.
     >             ( field_flag(ir, ju).ne.0 ) ) icount = icount +1
              if ( ( field_flag(ir, ju).ne.0 ).and.
     >             ( field_flag(ir, j ).ne.0 ) ) icount = icount +1
              if ( ( field_flag(ir, j ).ne.0 ).and.
     >             ( field_flag(ir, jd).ne.0 ) ) icount = icount +1
              if ( ( field_flag(ir, jd).ne.0 ).and.
     >             ( field_flag(i , jd).ne.0 ) ) icount = icount +1
              if ( ( field_flag(i , jd).ne.0 ).and.
     >             ( field_flag(il, jd).ne.0 ) ) icount = icount +1
              if ( ( field_flag(il, jd).ne.0 ).and.
     >             ( field_flag(il, j ).ne.0 ) ) icount = icount +1
              if ( ( field_flag(il, j ).ne.0 ).and.
     >             ( field_flag(il, ju).ne.0 ) ) icount = icount +1

c             Accept it if only one neighbour was found found
              if ( icount.eq.1 ) then
                 indx = i
                 indy = j
                 goto 110
              endif

            endif
        enddo
      enddo

c     Find a starting point for a frontal system: Ok, accept every point
c     which is not yet attributed to a frontal system
      indx = 0
      indy = 0
      do i=1,nx
        do j=1,ny
            if ( field_flag(i,j).ne.0 ) then
               indx = i
               indy = j
               goto 110
            endif
         enddo
      enddo

c     A next starting point (indx/indy) of a line was found
 110  continue

      if ( indx.eq.0 ) goto 140

c     Init the stack and track for next line segment
      nstack                = 0
      ntrack                = 0
      maxdist               = 0.
      trlen                 = 0
      icount                = 0

c     Sequentially extend the track
 120  continue

c     Put the point (indx,indy) onto the track; mark it as visited
      ntrack                = ntrack + 1
      trackx(ntrack)        = indx
      tracky(ntrack)        = indy
      field_flag(indx,indy) = 2

c     Set neighboring points for the current point on a line
      i0 = indx
      j0 = indy
      il = indx-1
      if (il.lt.1) il=nx
      ir = indx+1
      if ( ir.gt.nx ) ir=1
      jd = indy-1
      if (jd.lt.1) jd=1
      ju = indy+1
      if ( ju.gt.ny ) ju=ny

c     Reset flag for line continuation
      contflag = 0

c     try east : connected by grid
      i = ir
      j = j0
      if ( (field_flag(i,j).eq.1).and.(contflag.eq.0) ) then
           contflag        = 1
           indx            = i
           indy            = j
      elseif ( (field_flag(i,j).eq.1).and.(contflag.ne.0) ) then
           nstack          = nstack + 1
           stackx(nstack)  = i
           stacky(nstack)  = j
           stackp(nstack)  = ntrack
      endif

c     try north-east : connected by grid
      i = ir
      j = ju
      if ( (field_flag(i,j).eq.1).and.(contflag.eq.0) ) then
           contflag        = 1
           indx            = i
           indy            = j
      elseif ( (field_flag(i,j).eq.1).and.(contflag.ne.0) ) then
           nstack          = nstack + 1
           stackx(nstack)  = i
           stacky(nstack)  = j
           stackp(nstack)  = ntrack
      endif

c     try south-east : connected by grid
      i = ir
      j = jd
      if ( (field_flag(i,j).eq.1).and.(contflag.eq.0) ) then
           contflag        = 1
           indx            = i
           indy            = j
      elseif ( (field_flag(i,j).eq.1).and.(contflag.ne.0) ) then
           nstack          = nstack + 1
           stackx(nstack)  = i
           stacky(nstack)  = j
           stackp(nstack)  = ntrack
      endif

c     try north : connected by grid
      i = i0
      j = ju
      if ( (field_flag(i,j).eq.1).and.(contflag.eq.0) ) then
           contflag        = 1
           indx            = i
           indy            = j
      elseif ( (field_flag(i,j).eq.1).and.(contflag.ne.0) ) then
           nstack          = nstack + 1
           stackx(nstack)  = i
           stacky(nstack)  = j
           stackp(nstack)  = ntrack
      endif

c     try south : connected by grid
      i = i0
      j = jd
      if ( (field_flag(i,j).eq.1).and.(contflag.eq.0) ) then
           contflag        = 1
           indx            = i
           indy            = j
      elseif ( (field_flag(i,j).eq.1).and.(contflag.ne.0) ) then
           nstack          = nstack + 1
           stackx(nstack)  = i
           stacky(nstack)  = j
           stackp(nstack)  = ntrack
      endif

c     try west : connected by grid
      i = il
      j = j0
      if ( (field_flag(i,j).eq.1).and.(contflag.eq.0) ) then
           contflag        = 1
           indx            = i
           indy            = j
      elseif ( (field_flag(i,j).eq.1).and.(contflag.ne.0) ) then
           nstack          = nstack + 1
           stackx(nstack)  = i
           stacky(nstack)  = j
           stackp(nstack)  = ntrack
      endif

c     try north-west : connected by grid
      i = il
      j = ju
      if ( (field_flag(i,j).eq.1).and.(contflag.eq.0) ) then
           contflag        = 1
           indx            = i
           indy            = j
      elseif ( (field_flag(i,j).eq.1).and.(contflag.ne.0) ) then
           nstack          = nstack + 1
           stackx(nstack)  = i
           stacky(nstack)  = j
           stackp(nstack)  = ntrack
      endif

c     try south-west : connected by grid
      i = il
      j = jd
      if ( (field_flag(i,j).eq.1).and.(contflag.eq.0) ) then
           contflag        = 1
           indx            = i
           indy            = j
      elseif ( (field_flag(i,j).eq.1).and.(contflag.ne.0) ) then
           nstack          = nstack + 1
           stackx(nstack)  = i
           stacky(nstack)  = j
           stackp(nstack)  = ntrack
      endif

c     No grid neighbor was found; try to extend the line by building a bridge to
c     a nearby point - don't be critical in selection: take the first one which
c     fufills the distance criterion;
      flag = 0
      do i=1,nx
         do j=1,ny


           if ( ( field_flag(i,j).eq.1 ).and.
     >          ( cluster(i,j).ne.cluster(indx,indy) ) )
     >     then
              lon1 = xmin + real(i   -1) * dx
              lat1 = ymin + real(j   -1) * dy
              lon2 = xmin + real(indx-1) * dx
              lat2 = ymin + real(indy-1) * dy
              dist = sdis(lon1,lat1,lon2,lat2)
              if ( dist.le.inp_distance ) then
                 i0   = i
                 j0   = j
                 flag = 1
                 goto 130
              endif
           endif
         enddo
      enddo
 130  continue

c     If the bridge is still ok, connect it
      if ( ( flag.eq.1 ).and.( contflag.eq.0) ) then
           contflag              = 1
           indx                  = i0
           indy                  = j0
      elseif ( (flag.eq.1).and.(contflag.ne.0) ) then
           nstack          = nstack + 1
           stackx(nstack)  = i0
           stacky(nstack)  = j0
           stackp(nstack)  = ntrack
      endif

c     Write line segments to test file
      if ( test.eq.2 ) then

        print*,nstep,i0,j0,' -> ',contflag,indx,indy
        print*,'-----------------------------------------------------'
        do i=1,ntrack
            print*,i,trackx(i),tracky(i)
        enddo
        print*,'-----------------------------------------------------'
        print*
        cdfname=testfile
        varname='TRACK'
        nstep  = nstep + 1.
        if ( nstep.eq.1.) then
           call writecdf2D(cdfname,varname,field_flag,nstep,
     >                     dx,dy,xmin,ymin,nx,ny,1,1)
        else
           call writecdf2D(cdfname,varname,field_flag,nstep,
     >                     dx,dy,xmin,ymin,nx,ny,0,0)
        endif

      endif

c     Try to extend the line segment further
      if ( contflag.ne.0 ) goto 120

c     -----------------------------------------------------------------
c     The line segment is complete; decide whether to keep it (if it is
c     the longest segment), or whether to go back to the last branching
c     point and look for other possible extensions
c     -----------------------------------------------------------------

c     Count the number of variations - if the maximum number of
c     variations is reached, accept the so far found line segment
      icount = icount + 1
      if ( icount.gt.maxvar ) goto 135

c     Write complete line segments to test file
      if ( test.eq.3 ) then

        print*,'nstep ',nstep
        print*,'-----------------------------------------------------'
        do i=1,ntrack
            print*,i,trackx(i),tracky(i)
        enddo
        print*,'-----------------------------------------------------'
        print*
        cdfname=testfile
        varname='TRACK'
        nstep  = nstep + 1.
        if ( nstep.eq.1.) then
           call writecdf2D(cdfname,varname,field_flag,nstep,
     >                     dx,dy,xmin,ymin,nx,ny,1,1)
        else
           call writecdf2D(cdfname,varname,field_flag,nstep,
     >                     dx,dy,xmin,ymin,nx,ny,0,0)
        endif

      endif

c     The line is complete - determine its length
      dist = 0
      if ( ntrack.gt.1 ) then
        do i=1,ntrack-1
           lon1 = xmin + real( trackx(i  ) - 1) * dx
           lat1 = ymin + real( tracky(i  ) - 1) * dy
           lon2 = xmin + real( trackx(i+1) - 1) * dx
           lat2 = ymin + real( tracky(i+1) - 1) * dy
           dist = dist + sdis(lon1,lat1,lon2,lat2)
        enddo
      else
        dist = 0.
      endif

c     It is the longest path so far? - keep it
      if ( dist.ge.maxdist ) then
         do i=1,ntrack
           trlon(i) = xmin + real( trackx(i) - 1) * dx
           trlat(i) = ymin + real( tracky(i) - 1) * dy
         enddo
         trlen   = ntrack
         maxdist = dist
      endif

c     Make path free again for all points since last branching point
      do i=ntrack,stackp(nstack)+1,-1
	    i0                = trackx(i)
	    j0                = tracky(i)
	    field_flag(i0,j0) = 1
      enddo
      ntrack = stackp(nstack)

c     Set the new position and go back to last branching point
      if ( nstack.ge.1 ) then
         indx   = stackx(nstack)
         indy   = stacky(nstack)
         nstack = nstack -1
         goto 120
      endif

c     -----------------------------------------------------------------
c     All possible paths are checked from the starting point; the
c     longest path is saved in <trlon,trlat>. Accept it and prepare
c     field for search of next line
c     -----------------------------------------------------------------

 135  continue

c     Save the new line segment
      nsegm         = nsegm + 1
      isegm0(nsegm) = nline + 1
      do i=1,trlen
         nline = nline + 1
         lilon(nline) = trlon(i)
         lilat(nline) = trlat(i)
      enddo
      isegm1(nsegm)    = nline
      segm_stat(nsegm) = 1

c     Remove the final track from field - keep branching points
      do i=1,trlen
         indx = nint( ( trlon(i) -xmin ) / dx + 1 )
         indy = nint( ( trlat(i) -ymin ) / dy + 1 )
         field_flag(indx,indy) = 0
      enddo

c     Allow all other points to be visited again
      do i=1,nx
        do j=1,ny
           if ( field_flag(i,j).ne.0 ) field_flag(i,j) = 1
        enddo
      enddo

c     Look for the next line
      goto 100

c     All points are attributed to a path
 140  continue

c     Write line segments to test file
      if ( test.eq.1 ) then
        do i=1,nx
        do j=1,ny
           field_flag(i,j) = 0
        enddo
        enddo
        ilabel = 0
        do i=1,nsegm
         if ( segm_stat(i).eq.1 ) then
            ilabel = ilabel + 1
            do j=isegm0(i),isegm1(i)
               indx = nint( ( lilon(j)-xmin ) / dx + 1. )
               indy = nint( ( lilat(j)-ymin ) / dy + 1. )
               if ( (indx.ge.1).and.(indx.le.nx).and.
     >              (indy.ge.1).and.(indy.le.ny) )
     >         then
                 field_flag(indx,indy) = ilabel
               endif
            enddo
          endif
        enddo
        cdfname=testfile
        varname='SEGMENT'
        nstep  = 0.
        call writecdf2D(cdfname,varname,field_flag,nstep,
     >                  dx,dy,xmin,ymin,nx,ny,0,1)
      endif

c     -------------------------------------------------------------------
c     Combine lines at starting or ending point - so far the identification
c     of lines might have started in the middle of a frontal line
c     -------------------------------------------------------------------

      do n=1,niter

      i = 0
      j = 0
c     Check whether lines can be combined
      do while (i.le.nsegm)
         i      = i + 1
         j      = 0
         ilabel = 0

         do while (j.le.nsegm)
           j = j+1

c          Skip line sgements which have already been extended
c          Also avoid self-connection!
           if ( segm_stat(i).eq.0 ) goto 150
           if ( segm_stat(j).eq.0 ) goto 150
           if ( i.eq.j )            goto 150 

c          start of i with start of j
           lon1 = lilon( isegm0(i) )
           lat1 = lilat( isegm0(i) )
           lon2 = lilon( isegm0(j) )
           lat2 = lilat( isegm0(j) )
           dist = sdis( lon1,lat1,lon2,lat2 )
           if ( dist.le.inp_distance ) then
              nsegm         = nsegm + 1
              isegm0(nsegm) = nline + 1  
              do k=isegm1(i),isegm0(i),-1
                  nline        = nline + 1
                  lilon(nline) = lilon(k)
                  lilat(nline) = lilat(k)
              enddo
              do k=isegm0(j),isegm1(j)
                  nline        = nline + 1
                  lilon(nline) = lilon(k)
                  lilat(nline) = lilat(k)
              enddo
              isegm1(nsegm)    = nline
              segm_stat(nsegm) = 1
              segm_stat(i)     = 0
              segm_stat(j)     = 0
              ilabel           = j
              goto 150
           endif

c          start of i with end of j
           lon1 = lilon( isegm0(i) )
           lat1 = lilat( isegm0(i) )
           lon2 = lilon( isegm1(j) )
           lat2 = lilat( isegm1(j) )
           dist = sdis( lon1,lat1,lon2,lat2 )
           if ( dist.le.inp_distance ) then
              nsegm         = nsegm + 1
              isegm0(nsegm) = nline + 1  
              do k=isegm0(j),isegm1(j)
                  nline        = nline + 1
                  lilon(nline) = lilon(k)
                  lilat(nline) = lilat(k)
              enddo
              do k=isegm0(i),isegm1(i)
                  nline        = nline + 1
                  lilon(nline) = lilon(k)
                  lilat(nline) = lilat(k)
              enddo
              isegm1(nsegm)    = nline
              segm_stat(nsegm) = 1
              segm_stat(i)     = 0
              segm_stat(j)     = 0
              ilabel           = j
              goto 150
           endif

c          end of i with start of j
           lon1 = lilon( isegm1(i) )
           lat1 = lilat( isegm1(i) )
           lon2 = lilon( isegm0(j) )
           lat2 = lilat( isegm0(j) )
           dist = sdis( lon1,lat1,lon2,lat2 )
           if ( dist.le.inp_distance ) then
              nsegm         = nsegm + 1
              isegm0(nsegm) = nline + 1  
              do k=isegm0(i),isegm1(i)
                  nline        = nline + 1
                  lilon(nline) = lilon(k)
                  lilat(nline) = lilat(k)
              enddo
              do k=isegm0(j),isegm1(j)
                  nline        = nline + 1
                  lilon(nline) = lilon(k)
                  lilat(nline) = lilat(k)
              enddo
              isegm1(nsegm)    = nline
              segm_stat(nsegm) = 1
              segm_stat(i)     = 0
              segm_stat(j)     = 0
              ilabel           = j
              goto 150
           endif

c          end of i with end of j
           lon1 = lilon( isegm1(i) )
           lat1 = lilat( isegm1(i) )
           lon2 = lilon( isegm1(j) )
           lat2 = lilat( isegm1(j) )
           dist = sdis( lon1,lat1,lon2,lat2 )
           if ( dist.le.inp_distance ) then
              nsegm         = nsegm + 1
              isegm0(nsegm) = nline + 1  
              do k=isegm0(i),isegm1(i)
                  nline        = nline + 1
                  lilon(nline) = lilon(k)
                  lilat(nline) = lilat(k)
              enddo
              do k=isegm1(j),isegm0(j),-1
                  nline        = nline + 1
                  lilon(nline) = lilon(k)
                  lilat(nline) = lilat(k)
              enddo
              isegm1(nsegm)    = nline 
              segm_stat(nsegm) = 1
              segm_stat(i)     = 0
              segm_stat(j)     = 0
              ilabel           = j
              goto 150
           endif

c          Exit point for loop
 150       continue

        enddo

      enddo

      enddo

c     Write line segments to test file
      if ( test.eq.1 ) then
        do i=1,nx
        do j=1,ny
           field_flag(i,j) = 0
        enddo
        enddo
        ilabel = 0
        do i=1,nsegm
         if ( segm_stat(i).eq.1 ) then
            ilabel = ilabel + 1
            do j=isegm0(i),isegm1(i)
               indx = nint( ( lilon(j)-xmin ) / dx + 1. )
               indy = nint( ( lilat(j)-ymin ) / dy + 1. )
               if ( (indx.ge.1).and.(indx.le.nx).and.
     >              (indy.ge.1).and.(indy.le.ny) )
     >         then
                 field_flag(indx,indy) = ilabel
               endif
            enddo
          endif
        enddo
        cdfname=testfile
        varname='LINES'
        nstep  = 0.
        call writecdf2D(cdfname,varname,field_flag,nstep,
     >                  dx,dy,xmin,ymin,nx,ny,0,1)
      endif

c     -------------------------------------------------------------------
c     Apply a smoothing cubic spline to frontal line
c     -------------------------------------------------------------------

      if ( smoothing.ne.0 ) then

c       Set the smoothing parameter
        smooth = real(smoothing)

c       Avoid jumps of longitude at date line
        do i=1,nsegm
           if ( segm_stat(i).eq.1 )then

           do j=isegm0(i)+1,isegm1(i)
             d1 = abs(  lilon(j)       - lilon(j-1) )
             d2 = abs( (lilon(j)+360.) - lilon(j-1) )
             d3 = abs( (lilon(j)-360.) - lilon(j-1) )
             if ( (d2.lt.d1).and.(d2.lt.d3) ) then
                lilon(j) = lilon(j)+360.
             else if ( (d3.lt.d1).and.(d3.lt.d1) ) then
                lilon(j) = lilon(j)-360.
             endif
         enddo

         endif
        enddo

c       Loop over all lines
        do i=1,nsegm

c        Check whether it is a valid line
         if ( segm_stat(i).eq.0 ) goto 160

c        Write the line to screen
         if ( test.eq.5 ) then
           do j=isegm0(i),isegm1(i)
             write(*,'(2f10.2)') lilon(j),lilat(j)
           enddo
         endif

c        Cubic-spline smoother for longitude
         iopt = 0
         m    = 0
         do j=isegm0(i),isegm1(i)
            m = m + 1
            x(m) = real(m) - 1.
            y(m) = lilon(j)
            w(m) = 0.5
         enddo
         do j=1,m
            x(j) = ( x(j) - x(1) ) / ( x(m) - x(1) )
         enddo
         xb   = x(1)
         xe   = x(m)
         k    = 3
         s    = smooth
         nest = m+k+1
         lwrk = maxlen

         call curfit(iopt,m,x,y,w,x(1),x(m),3,s,m+k+1,n_lon,
     >               t_lon,c_lon,fp_lon,wrk,lwrk,iwrk,ier)
         if ( ier.gt.0 ) then
            segm_stat(i) = 0
            goto 160
         endif

c        Cubic-spline smoother for latitude
         iopt = 0
         m    = 0
         do j=isegm0(i),isegm1(i)
            m = m + 1
            x(m) = real(m) - 1.
            y(m) = lilat(j)
            w(m) = 1.0
         enddo
         do j=1,m
            x(j) = ( x(j) - x(1) ) / ( x(m) - x(1) )
         enddo
         xb   = x(1)
         xe   = x(m)
         k    = 3
         s    = smooth
         nest = m+k+1
         lwrk = maxlen

         call curfit(iopt,m,x,y,w,x(1),x(m),3,s,m+k+1,n_lat,
     >               t_lat,c_lat,fp_lat,wrk,lwrk,iwrk,ier)
         if ( ier.gt.0 ) then
            segm_stat(i) = 0
            goto 160
         endif

c        Apply the cubic spline
         do j=1,m
            x(j) = (real(j) - 1.) / ( real(m) - 1.)
         enddo
         call splev(t_lon,n_lon,c_lon,3,x,y_lon,m,ier)
         if ( ier.gt.0 ) then
            segm_stat(i) = 0
            goto 160
         endif
         call splev(t_lat,n_lat,c_lat,3,x,y_lat,m,ier)
         if ( ier.gt.0 ) then
            segm_stat(i) = 0
            goto 160
         endif

c        Save the new line - everything is fine, keep it
         m = 0
         do j=isegm0(i),isegm1(i)
            m = m + 1
            lilon(j) = y_lon(m)
            lilat(j) = y_lat(m)
         enddo

c       Exit point for loop over all lines
160     continue

       enddo


c       Bring all longitudes back into data domain
        do i=1,nsegm
         if ( segm_stat(i).eq.1 ) then
             do j=isegm0(i),isegm1(i)
                if ( lilon(j).gt.xmax ) lilon(j) = lilon(j) - 360.
                if ( lilon(j).lt.xmin ) lilon(j) = lilon(j) + 360.
             enddo
          endif
        enddo

c       Write smoothed lines to test file
        if ( test.eq.1 ) then
          do i=1,nx
          do j=1,ny
           field_flag(i,j) = 0
          enddo
          enddo
          do i=1,nsegm
           if ( segm_stat(i).eq.1 ) then
            do j=isegm0(i),isegm1(i)
               indx = nint( ( lilon(j)-xmin ) / dx + 1. )
               indy = nint( ( lilat(j)-ymin ) / dy + 1. )
               if ( (indx.ge.1).and.(indx.le.nx).and.
     >              (indy.ge.1).and.(indy.le.ny) )
     >         then
                 field_flag(indx,indy) = 1
               endif
            enddo
           endif
          enddo
          cdfname=testfile
          varname='SMOOTHING'
          nstep  = 0.
          call writecdf2D(cdfname,varname,field_flag,nstep,
     >                  dx,dy,xmin,ymin,nx,ny,0,1)
        endif

      endif

c     -------------------------------------------------------------------
c     Remove all lines which are too short
c     -------------------------------------------------------------------

c     Get the length of all lines
      do i=1,nsegm

         if ( segm_stat(i).eq.0 ) goto 170

         segm_length(i) = 0.
         do j=isegm0(i),isegm1(i)-1
             lon1 = lilon( j   )
             lat1 = lilat( j   )
             lon2 = lilon( j+1 )
             lat2 = lilat( j+1 )
             dist = sdis( lon1,lat1,lon2,lat2 )
             segm_length(i) = segm_length(i) + dist
         enddo

c        Exit point for loop
 170     continue

      enddo

c     Mark segments for deleting which are too short
      do i=1,nsegm
        if ( (segm_stat(i).eq.1).and.
     >       (segm_length(i).le.inp_minlength) )
     >  then
            segm_stat(i) = 0
        endif
      enddo

c     Write system length filtered lines to test file
      if ( test.eq.1 ) then
        do i=1,nx
        do j=1,ny
           field_flag(i,j) = 0
        enddo
        enddo
        do i=1,nsegm
         if ( segm_stat(i).eq.1 ) then
            do j=isegm0(i),isegm1(i)
               indx = nint( ( lilon(j)-xmin ) / dx + 1. )
               indy = nint( ( lilat(j)-ymin ) / dy + 1. )
               if ( (indx.ge.1).and.(indx.le.nx).and.
     >              (indy.ge.1).and.(indy.le.ny) )
     >         then
                 field_flag(indx,indy) = 1
               endif
            enddo
          endif
        enddo
        cdfname=testfile
        varname='LENGTH'
        nstep  = 0.
        call writecdf2D(cdfname,varname,field_flag,nstep,
     >                  dx,dy,xmin,ymin,nx,ny,0,1)
      endif

c     -------------------------------------------------------------------
c     Attribute each frontal line to a frontal system
c     -------------------------------------------------------------------

c     Reset the array with the systems labels, and reset the counter for
c     different  frontal systems
      do i=1,nsegm
        segm_sysid(i) = 0
      enddo
      icount = 0

c     Loop over all line segments
      do i=1,nsegm
        do j=1,nsegm

c          Check that both lines are valid
           if ( segm_stat(i).eq.0 ) goto 145
           if ( segm_stat(j).eq.0 ) goto 145
           if ( i.eq.j )            goto 145

c          Now look for close encounters, get the minimum distance
           mindist = -1.
           do k=isegm0(i),isegm1(i)
           do l=isegm0(j),isegm1(j)

             lon1 = lilon( k )
             lat1 = lilat( k )
             lon2 = lilon( l )
             lat2 = lilat( l )
             dist = sdis( lon1,lat1,lon2,lat2 )
             if ( mindist.lt.0. ) then
                mindist = dist
             elseif ( dist.lt.mindist ) then
                mindist = dist
             endif

           enddo
           enddo

c          The two lines are neighbours - attribuet to both of them
c          a unique system label; make sure that this label is also
c          exported to all other neighbouring segments
           if ( mindist.gt.inp_distance ) goto 145

c             Both lines have no system label so far - create a new one
              if ( (segm_sysid(i).eq.0).and.
     >             (segm_sysid(j).eq.0) )
     >        then
                 icount = icount + 1
                 segm_sysid(i) = icount
                 segm_sysid(j) = icount

c             Line j already has a label -> export it to i
              elseif ( (segm_sysid(i).eq.0).and.
     >                 (segm_sysid(j).ne.0) )
     >        then
                  segm_sysid(i) = segm_sysid(j)

c             Line i already has a label -> export it to j
              elseif ( (segm_sysid(i).ne.0).and.
     >                 (segm_sysid(j).eq.0) )
     >        then
                  segm_sysid(j) = segm_sysid(i)

c             Both lines have a label - combine them
              else
                 do k=1,nsegm
                   if ( segm_sysid(k).eq.segm_sysid(j) ) then
                      segm_sysid(k) = segm_sysid(i)
                   endif
                 enddo
                 segm_sysid(j) = segm_sysid(i)

              endif

c          Exit point for loop over all line combinations
145        continue

         enddo
      enddo

c     Now attribute system label to all lines which are single
      do i=1,nsegm
        if ( (segm_stat(i).eq.1).and.(segm_sysid(i).eq.0) ) then
           icount = icount +1
           segm_sysid(i) = icount
        endif
      enddo

c     Write system label to test file
      if ( test.eq.1 ) then
        do i=1,nx
        do j=1,ny
           field_flag(i,j) = 0
        enddo
        enddo
        do i=1,nsegm
         if ( segm_stat(i).eq.1 ) then
            do j=isegm0(i),isegm1(i)
               indx = nint( ( lilon(j)-xmin ) / dx + 1. )
               indy = nint( ( lilat(j)-ymin ) / dy + 1. )
               if ( (indx.ge.1).and.(indx.le.nx).and.
     >              (indy.ge.1).and.(indy.le.ny) )
     >         then
                 field_flag(indx,indy) = segm_sysid(i)
               endif
            enddo
          endif
        enddo
        cdfname=testfile
        varname='SYSTEM'
        nstep  = 0.
        call writecdf2D(cdfname,varname,field_flag,nstep,
     >                  dx,dy,xmin,ymin,nx,ny,0,1)
      endif

c     -------------------------------------------------------------------
c     Reparamerterize the frontal lines - equidistant spacing
c     The spacing is passed as parameter <reparamm>. If <reparam=0>, no
c     reparameterization is applied.
c     -------------------------------------------------------------------

      if ( reparam.ne.0 ) then

c     Avoid jumps of longitude at date line
      do i=1,nsegm
         if ( segm_stat(i).eq.1 )then

           do j=isegm0(i)+1,isegm1(i)
             d1 = abs(  lilon(j)       - lilon(j-1) )
             d2 = abs( (lilon(j)+360.) - lilon(j-1) )
             d3 = abs( (lilon(j)-360.) - lilon(j-1) )
             if ( (d2.lt.d1).and.(d2.lt.d3) ) then
                lilon(j) = lilon(j)+360.
             else if ( (d3.lt.d1).and.(d3.lt.d1) ) then
                lilon(j) = lilon(j)-360.
             endif
         enddo

        endif
      enddo

c     Loop over all lines
      do i=1,nsegm

c        Skip invalid lines
         if ( segm_stat(i).eq.0 ) goto 180

c        Cubic-spline for longitude
         iopt = 0
         m    = 0
         do j=isegm0(i),isegm1(i)
            m = m + 1
            if ( m.eq.1 ) then
                x(m) = 0.
            else
                lon1    = lilon( j   )
                lat1    = lilat( j   )
                lon2    = lilon( j-1 )
                lat2    = lilat( j-1 )
                x(m)    = x(m-1) + sdis( lon1,lat1,lon2,lat2 )
            endif
            y(m) = lilon(j)
            w(m) = 0.5
            if ( m.ne.1 ) then
               if ( abs(x(m)-x(m-1)).lt.eps ) m = m-1
            endif
         enddo
         xb   = x(1)
         xe   = x(m)
         k    = 3
         s    = 0.
         nest = m+k+1
         lwrk = maxlen

         call curfit(iopt,m,x,y,w,x(1),x(m),3,s,m+k+1,n_lon,
     >               t_lon,c_lon,fp_lon,wrk,lwrk,iwrk,ier)
         if ( ier.gt.0 ) then
            print*,'Removing line ',i,ier
            segm_stat(i) = 0
            goto 180
         endif

c        Cubic-spline smoother for latitude
         iopt = 0
         m    = 0
         do j=isegm0(i),isegm1(i)
            m = m + 1
            if ( m.eq.1 ) then
                x(m) = 0.
            else
                lon1    = lilon( j   )
                lat1    = lilat( j   )
                lon2    = lilon( j-1 )
                lat2    = lilat( j-1 )
                x(m)    = x(m-1) + sdis( lon1,lat1,lon2,lat2 )
            endif
            y(m) = lilat(j)
            w(m) = 1.0
            if ( m.ne.1 ) then
               if ( abs(x(m)-x(m-1)).lt.eps ) m = m-1
            endif
         enddo
         xb   = x(1)
         xe   = x(m)
         k    = 3
         s    = 0.
         nest = m+k+1
         lwrk = maxlen

         call curfit(iopt,m,x,y,w,x(1),x(m),3,s,m+k+1,n_lat,
     >               t_lat,c_lat,fp_lat,wrk,lwrk,iwrk,ier)
         if ( ier.gt.0 ) then
            print*,'Removing line ',i,ier
            segm_stat(i) = 0
            goto 180
         endif

c        Apply the cubic spline - equidistant positions
         length = x(m)
         m = nint( length/real(reparam) )
         do j=1,m
            x(j) = real(j-1)/real(m-1) * length
         enddo
         call splev(t_lon,n_lon,c_lon,3,x,y_lon,m,ier)
         if ( ier.gt.0 ) then
            segm_stat(i) = 0
            goto 180
         endif
         call splev(t_lat,n_lat,c_lat,3,x,y_lat,m,ier)
         if ( ier.gt.0 ) then
            segm_stat(i) = 0
            goto 180
         endif

c        Check whether the vertices are now equidistantly separated
         if ( test.eq.5 ) then
            do j=1,m-1
              lon1      = y_lon( j   )
              lat1      = y_lat( j   )
              lon2      = y_lon( j+1 )
              lat2      = y_lat( j+1 )
              print*,x(j),y_lon(j),y_lat(j),sdis( lon1,lat1,lon2,lat2 )
           enddo
           print*,'next'
           read*,ch
        endif

c       Remember the new line
        nsegm            = nsegm + 1
        segm_stat(nsegm) = 1
        isegm0(nsegm)    = nline + 1
        do j=1,m
          nline        = nline + 1
          lilon(nline) = y_lon(j)
          lilat(nline) = y_lat(j)
        enddo
        isegm1(nsegm)      = nline
        segm_length(nsegm) = segm_length(i)
        segm_sysid(nsegm)  = segm_sysid(i)
        segm_stat(i)       = 0

c       Exit point for loop
 180    continue

      enddo

c     Bring all longitudes back into data domain
      do i=1,nsegm
         if ( segm_stat(i).eq.1 ) then
             do j=isegm0(i),isegm1(i)
                if ( lilon(j).gt.xmax ) lilon(j) = lilon(j) - 360.
                if ( lilon(j).lt.xmin ) lilon(j) = lilon(j) + 360.
             enddo
          endif
      enddo

c     Write reparameterized lines to test file
      if ( test.eq.1 ) then
        do i=1,nx
        do j=1,ny
           field_flag(i,j) = 0
        enddo
        enddo
        do i=1,nsegm
         if ( segm_stat(i).eq.1 ) then
            do j=isegm0(i),isegm1(i)
               indx = nint( ( lilon(j)-xmin ) / dx + 1. )
               indy = nint( ( lilat(j)-ymin ) / dy + 1. )
               if ( (indx.ge.1).and.(indx.le.nx).and.
     >              (indy.ge.1).and.(indy.le.ny) )
     >         then
                 field_flag(indx,indy) = 1
               endif
            enddo
          endif
        enddo
        cdfname=testfile
        varname='REPARAM'
        nstep  = 0.
        call writecdf2D(cdfname,varname,field_flag,nstep,
     >                  dx,dy,xmin,ymin,nx,ny,0,1)
      endif

      endif

c     -------------------------------------------------------------------
c     Define new THE and VEL fields
c     -------------------------------------------------------------------

c     Allow several extension steps
      do n=1,nexten

c     Copy data arrays to temporary arrays
      do i=1,nx
        do j=1,ny
            field_out1(i,j) = field_the (i,j)
            field_out2(i,j) = field_fvel(i,j)
        enddo
      enddo

c     Extend the arrays
      do i=1,nx
      do j=1,ny

c     Nothing to do if the field is already defined on grid point
      if ( abs(field_the(i,j)-mdv).lt.eps ) goto 190

c     Set neighboring points for the current point on a line
      il = i-1
      if (il.lt.1) il=nx
      ir = i+1
      if ( ir.gt.nx ) ir=1
      jd = j-1
      if (jd.lt.1) jd=1
      ju = j+1
      if ( ju.gt.ny ) ju=ny

c     Fill value for nearest neighbors - horizontal and vertical line
      if ( abs(field_the(il,j)-mdv).lt.eps ) then
         field_out1(il,j) = field_the (i,j)
         field_out2(il,j) = field_fvel(i,j)
      endif
      if ( abs(field_the(ir,j)-mdv).lt.eps ) then
         field_out1(ir,j) = field_the (i,j)
         field_out2(ir,j) = field_fvel(i,j)
      endif
      if ( abs(field_the(i, ju)-mdv).lt.eps ) then
         field_out1(i ,ju) = field_the (i,j)
         field_out2(i ,ju) = field_fvel(i,j)
      endif
      if ( abs(field_the(i, jd)-mdv).lt.eps ) then
         field_out1(i ,jd) = field_the (i,j)
         field_out2(i ,jd) = field_fvel(i,j)
      endif

c     Fill value for nearest neighbors - diagonal line
      if ( ( abs(field_the (il,ju)-mdv).lt.eps ).and.
     >     ( abs(field_out1(il,ju)-mdv).lt.eps ) )
     >then
         field_out1(il,ju) = field_the (i,j)
         field_out2(il,ju) = field_fvel(i,j)
      endif
      if ( ( abs(field_the (ir,ju)-mdv).lt.eps ).and.
     >     ( abs(field_out1(ir,ju)-mdv).lt.eps ) )
     >then
         field_out1(ir,ju) = field_the (i,j)
         field_out2(ir,ju) = field_fvel(i,j)
      endif
      if ( ( abs(field_the (il,ju)-mdv).lt.eps ).and.
     >     ( abs(field_out1(il,ju)-mdv).lt.eps ) )
     >then
         field_out1(il,ju) = field_the (i,j)
         field_out2(il,ju) = field_fvel(i,j)
      endif
      if ( ( abs(field_the (ir,jd)-mdv).lt.eps ).and.
     >     ( abs(field_out1(ir,jd)-mdv).lt.eps ) )
     >then
         field_out1(ir,jd) = field_the (i,j)
         field_out2(ir,jd) = field_fvel(i,j)
      endif

c     Exit point for loop
190   continue

      enddo
      enddo

c     Copy remporary arrays back to data arrays
      do i=1,nx
        do j=1,ny
            field_the (i,j) = field_out1(i,j)
            field_fvel(i,j) = field_out2(i,j)
        enddo
      enddo

      enddo

c     Write extended field to test file
      if ( test.eq.1 ) then
        cdfname=testfile
        varname=fieldname
        nstep  = 0.
        call writecdf2Dr(cdfname,varname,field_the,nstep,
     >                  dx,dy,xmin,ymin,nx,ny,0,1)
        cdfname=testfile
        varname='FVEL'
        nstep  = 0.
        call writecdf2Dr(cdfname,varname,field_fvel,nstep,
     >                  dx,dy,xmin,ymin,nx,ny,0,1)
      endif

c     -------------------------------------------------------------------
c     Write output - note that the valid segments/lines have segm_stat=1
c     -------------------------------------------------------------------

c     Write original fields to output
      cdfname=trim(out_filename)//'.cdf'
      varname='INPUT'
      call writecdf2D(cdfname,varname,field_inp,0.,
     >                   dx,dy,xmin,ymin,nx,ny,1,1)

c     Bring all longitudes back into data domain
      do i=1,nsegm
         if ( segm_stat(i).eq.1 ) then
             do j=isegm0(i),isegm1(i)
                if ( lilon(j).gt.xmax ) lilon(j) = lilon(j) - 360.
                if ( lilon(j).lt.xmin ) lilon(j) = lilon(j) + 360.
             enddo
          endif
      enddo

c     Bring frontal lines to grid
      do i=1,nx
        do j=1,ny
           field_flag(i,j) = 0
        enddo
      enddo
      ilabel = 0
      do i=1,nsegm
         if ( segm_stat(i).eq.1 ) then
            ilabel = ilabel + 1
            do j=isegm0(i),isegm1(i)
               indx = nint( ( lilon(j)-xmin ) / dx + 1. )
               indy = nint( ( lilat(j)-ymin ) / dy + 1. )
               if ( (indx.ge.1).and.(indx.le.nx).and.
     >              (indy.ge.1).and.(indy.le.ny) )
     >         then
                 field_flag(indx,indy) = ilabel
               endif
            enddo
          endif
      enddo

      cdfname=trim(out_filename)//'.cdf'
      varname='LINE'
      call writecdf2D(cdfname,varname,field_flag,0.,
     >                   dx,dy,xmin,ymin,nx,ny,0,1)

c     Bring frontal system to grid
      do i=1,nx
        do j=1,ny
           field_flag(i,j) = 0
        enddo
      enddo
      ilabel = 0
      do i=1,nsegm
         if ( segm_stat(i).eq.1 ) then
            ilabel = ilabel + 1
            do j=isegm0(i),isegm1(i)
               indx = nint( ( lilon(j)-xmin ) / dx + 1. )
               indy = nint( ( lilat(j)-ymin ) / dy + 1. )
               if ( (indx.ge.1).and.(indx.le.nx).and.
     >              (indy.ge.1).and.(indy.le.ny) )
     >         then
                 field_flag(indx,indy) = segm_sysid(i)
               endif
            enddo
          endif
      enddo

      cdfname=trim(out_filename)//'.cdf'
      varname='SYSTEM'
      call writecdf2D(cdfname,varname,field_flag,0.,
     >                   dx,dy,xmin,ymin,nx,ny,0,1)

c     Write THE gradient and frontal velocity
      do i=1,nx
        do j=1,ny
            if ( field_flag(i,j).eq.0 ) then
               field_the(i,j)  = mdv
               field_fvel(i,j) = mdv
            elseif ( ( field_flag(i,j).ne.0 ).and.
     >             ( abs(field_the(il,ju)-mdv).lt.eps ) ) then
                print*,'No value attributed to THE and FVEL',i,j
            endif
         enddo
      enddo
      cdfname=trim(out_filename)//'.cdf'
      varname=fieldname
      call writecdf2Dr(cdfname,varname,field_the,0.,
     >                 dx,dy,xmin,ymin,nx,ny,0,1)

      cdfname=trim(out_filename)//'.cdf'
      varname='FVEL'
      call writecdf2Dr(cdfname,varname,field_fvel,0.,
     >                 dx,dy,xmin,ymin,nx,ny,0,1)

c     Write frontal lines to ASCII file
      open(10,file=trim(out_filename)//'.table')

      ilabel = 0
      do i=1,nsegm
         if ( segm_stat(i).eq.1 ) then
            ilabel = ilabel + 1
            do j=isegm0(i),isegm1(i)
               write(10,'(i4,2f10.2,i4)') ilabel,lilon(j),lilat(j),
     >                                    segm_sysid(i)
            enddo
         endif
      enddo

C     Write front characteristics to ASCII table
      open(10,file=trim(out_filename)//'.char')

      ilabel = 0
      do i=1,nsegm
         if ( segm_stat(i).eq.1 ) then
            ilabel = ilabel + 1
               write(10,'(i4,f10.2)') ilabel,segm_length(i)
         endif
      enddo

      close(10)

c     -------------------------------------------------------------------
c     Exception handling
c     -------------------------------------------------------------------

      stop

 998  print*,'Problem with input netCDF file'
      stop

      end


c     *************************************************************************
c     * SUBROUTINE SECTION: GENERAL                                           *
c     *************************************************************************

c     ----------------------------------------------------------------
c     Spherical distance
c     ----------------------------------------------------------------

      real function sdis(xp,yp,xq,yq)
c
      real      re
      parameter (re=6370.)
      real      xp,yp,xq,yq,arg

      arg=sind(yp)*sind(yq)+cosd(yp)*cosd(yq)*cosd(xp-xq)
      if (arg.lt.-1.) arg=-1.
      if (arg.gt.1.) arg=1.
      sdis=re*acos(arg)

      end

c     *************************************************************************
c     * SUBROUTINE SECTION : NETCDF-IO                                        *
c     *************************************************************************

c     -----------------------------------------------
c     Subroutines to write integer field
c     -----------------------------------------------

      subroutine writecdf2D(cdfname,
     >                      varname,iarr,time,
     >                      dx,dy,xmin,ymin,nx,ny,
     >                      crefile,crevar)

c     Create and write to the netcdf file <cdfname>. The variable
c     with name <varname> and with time <time> is written. The data
c     are in the two-dimensional array <arr>. The list <dx,dy,xmin,
c     ymin,nx,ny> specifies the output grid. The flags <crefile,crevar> 
C     indicate whether the file/variable exist

      IMPLICIT NONE

c     Declaration of input parameters
      character*80 cdfname,varname
      integer      nx,ny
      integer      iarr(nx,ny)
      real         dx,dy,xmin,ymin
      real         time
      integer      crefile,crevar
            
c     Further variables
      real         varmin(4),varmax(4),stag(4)
      integer      ierr,cdfid,ndim,vardim(4)
      character*80 cstname
      real         mdv
      integer      i,j
      real         rarr(nx,ny)
     
c     Bring integer to real array
      do i=1,nx
        do j=1,ny
            rarr(i,j) = iarr(i,j)
        enddo
      enddo

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

C     If necessary, create the netcdf file
      if (crefile.ne.0) then
         call crecdf(cdfname,cdfid,
     >        varmin,varmax,ndim,cstname,ierr)
         if (ierr.ne.0) goto 903 
      endif

c     Check whether the variable is already defined on the netcdf file      
      if (crevar.ne.0) then
         call cdfwopn(cdfname,cdfid,ierr)
         call putdef(cdfid,varname,ndim,mdv,vardim,varmin,varmax,
     >        stag,ierr)
         if (ierr.ne.0) goto 904
         call clscdf(cdfid,ierr)
      endif

c     Write the data to the netcdf file, and close the file
      call cdfwopn(cdfname,cdfid,ierr)
      call putdat(cdfid,varname,time,1,rarr,ierr)
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

      END

c     -----------------------------------------------
c     Subroutines to write real field
c     -----------------------------------------------

      subroutine writecdf2Dr(cdfname,
     >                       varname,rarr,time,
     >                       dx,dy,xmin,ymin,nx,ny,
     >                       crefile,crevar)

c     Create and write to the netcdf file <cdfname>. The variable
c     with name <varname> and with time <time> is written. The data
c     are in the two-dimensional array <arr>. The list <dx,dy,xmin,
c     ymin,nx,ny> specifies the output grid. The flags <crefile,crevar>
C     indicate whether the file/variable exist

      IMPLICIT NONE

c     Declaration of input parameters
      character*80 cdfname,varname
      integer      nx,ny
      real         rarr(nx,ny)
      real         dx,dy,xmin,ymin
      real         time
      integer      crefile,crevar

c     Further variables
      real         varmin(4),varmax(4),stag(4)
      integer      ierr,cdfid,ndim,vardim(4)
      character*80 cstname
      real         mdv
      integer      i,j

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

C     If necessary, create the netcdf file
      if (crefile.ne.0) then
         call crecdf(cdfname,cdfid,
     >        varmin,varmax,ndim,cstname,ierr)
         if (ierr.ne.0) goto 903
      endif

c     Check whether the variable is already defined on the netcdf file
      if (crevar.ne.0) then
         call cdfwopn(cdfname,cdfid,ierr)
         call putdef(cdfid,varname,ndim,mdv,vardim,varmin,varmax,
     >        stag,ierr)
         if (ierr.ne.0) goto 904
         call clscdf(cdfid,ierr)
      endif

c     Write the data to the netcdf file, and close the file
      call cdfwopn(cdfname,cdfid,ierr)
      call putdat(cdfid,varname,time,1,rarr,ierr)
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

      END


c     ----------------------------------------------------------------
c     Check whether variable is found on netcdf file
c     ----------------------------------------------------------------

      subroutine check_varok (isok,varname,varlist,nvars)

c     Check whether the variable <varname> is in the list <varlist(nvars)>. 
c     If this is the case, <isok> is incremented by 1. Otherwise <isok> 
c     keeps its value.

      implicit none

c     Declaraion of subroutine parameters
      integer      isok
      integer      nvars
      character*80 varname
      character*80 varlist(nvars)

c     Auxiliary variables
      integer      i

c     Main
      do i=1,nvars
         if (trim(varname).eq.trim(varlist(i))) isok=isok+1
      enddo

      end

c     ----------------------------------------------------------------
c     Do a 2d clustering
c     ----------------------------------------------------------------

      subroutine clustering(outar,ntot,inar,nx,ny)

c     Given the input array <inar> with 0/1 entries (for example the
c     NCOFF array), the individual events are counted and each event
c     gets an individual 'stamp' which is specific to this event. An
c     event is characterised by 1-entries in <inar> which are connected.
c     <nx>,<ny> give the dimension of the arrays, <ntot> the total
c     number of events. It is assumed that the domain is periodic in
c     x direction.

      IMPLICIT NONE

c     Declaration of subroutine parameters
      integer nx,ny
      integer inar(nx,ny)
      integer outar(nx,ny)
      integer ntot

c     Auxiliary variables
      integer i,j

c     Copy inar to outar
      do i=1,nx
         do j=1,ny
            outar(i,j)=inar(i,j)
         enddo
      enddo

c     Do the clustering (based on simple connectivivity)
      ntot=1
      do i=1,nx
      do j=1,ny

c        Find an element which does not yet belong to a cluster
         if (outar(i,j).eq.1) then
            ntot=ntot+1
            outar(i,j)=ntot
            call connected_2d(outar,ntot,i,j,nx,ny)
         endif

      enddo
      enddo

c     Correct the output array (so far, the index is 1 unit to large)
      do i=1,nx
         do j=1,ny
            if (outar(i,j).gt.1) outar(i,j)=outar(i,j)-1
         enddo
      enddo
      ntot=ntot-1

      return

      END


      subroutine connected_2d (outar,ntot,i,j,nx,ny)

c     Mark all elements in outar which are connected to element i,j

c     Declaration of subroutine parameters
      integer nx,ny
      integer ntot
      integer i,j
      integer outar(nx,ny)

c     Auxiliary variables
      integer   il,ir,ju,jd,im,jm
      integer   k
      integer   stack
      integer   nmax
      parameter (nmax=5000000)
      integer   indx(nmax),indy(nmax)

c     Push the first element on the stack
      stack=1
      indx(stack)=i
      indy(stack)=j

c     Define the indices of the neighbouring elements
 100  continue
      il=indx(stack)-1
      if (il.eq.0) il=nx
      ir=indx(stack)+1
      if (ir.eq.(nx+1)) ir=1
      ju=indy(stack)+1
      jd=indy(stacK)-1

      if (ju.eq.(ny+1)) then
         stack=stack-1
         goto 200
      else
         im=indx(stack)
         jm=indy(stack)
         stack=stack-1
      endif

      if (stack.gt.(nmax-nx)) then
         print*,'Stack overflow in clustering'
         stop
      endif

c     Mark all connected elements (build up the stack)
      if ((ju.ne.(ny+1)).and.(jd.ne.0)) then
         if (outar(il,jm).eq.1) then
            outar(il,jm)=ntot
            stack=stack+1
            indx(stack)=il
            indy(stack)=jm
         endif
         if (outar(ir,jm).eq.1) then
            outar(ir,jm)=ntot
            stack=stack+1
            indx(stack)=ir
            indy(stack)=jm
         endif
         if (outar(im,ju).eq.1) then
            outar(im,ju)=ntot
            stack=stack+1
            indx(stack)=im
            indy(stack)=ju
         endif
         if (outar(im,jd).eq.1) then
            outar(im,jd)=ntot
            stack=stack+1
            indx(stack)=im
            indy(stack)=jd
         endif
         if (outar(il,jd).eq.1) then
            outar(il,jd)=ntot
            stack=stack+1
            indx(stack)=il
            indy(stack)=jd
         endif
         if (outar(il,ju).eq.1) then
            outar(il,ju)=ntot
            stack=stack+1
            indx(stack)=il
            indy(stack)=ju
         endif
         if (outar(ir,jd).eq.1) then
            outar(ir,jd)=ntot
            stack=stack+1
            indx(stack)=ir
            indy(stack)=jd
         endif
         if (outar(ir,ju).eq.1) then
            outar(ir,ju)=ntot
            stack=stack+1
            indx(stack)=ir
            indy(stack)=ju
         endif

      else if (jd.eq.0) then
          if (outar(il,jm).eq.1) then
            outar(il,jm)=ntot
            stack=stack+1
            indx(stack)=il
            indy(stack)=jm
         endif
         if (outar(ir,jm).eq.1) then
            outar(ir,jm)=ntot
            stack=stack+1
            indx(stack)=ir
            indy(stack)=jm
         endif
         if (outar(im,ju).eq.1) then
            outar(im,ju)=ntot
            stack=stack+1
            indx(stack)=im
            indy(stack)=ju
         endif
         if (outar(il,ju).eq.1) then
            outar(il,ju)=ntot
            stack=stack+1
            indx(stack)=il
            indy(stack)=ju
         endif
         if (outar(ir,ju).eq.1) then
            outar(ir,ju)=ntot
            stack=stack+1
            indx(stack)=ir
            indy(stack)=ju
         endif

      else if (ju.eq.(ny+1)) then
         if (outar(il,jm).eq.1) then
            outar(il,jm)=ntot
            stack=stack+1
            indx(stack)=il
            indy(stack)=jm
         endif
         if (outar(ir,jm).eq.1) then
            outar(ir,jm)=ntot
            stack=stack+1
            indx(stack)=ir
            indy(stack)=jm
         endif
         if (outar(im,jd).eq.1) then
            outar(im,jd)=ntot
            stack=stack+1
            indx(stack)=im
            indy(stack)=jd
         endif
         if (outar(il,jd).eq.1) then
            outar(il,jd)=ntot
            stack=stack+1
            indx(stack)=il
            indy(stack)=jd
         endif
         if (outar(ir,jd).eq.1) then
            outar(ir,jd)=ntot
            stack=stack+1
            indx(stack)=ir
            indy(stack)=jd
         endif
         do k=1,nx
            outar(k,jm)=ntot
            stack=stack+1
            indx(stack)=k
            indy(stack)=jm
         enddo
      endif

 200  continue
      if (stack.gt.0) goto 100

      return

      end





